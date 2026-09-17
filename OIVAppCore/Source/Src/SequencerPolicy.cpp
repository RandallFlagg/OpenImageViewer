#include <OIVAppCore/SequencerPolicy.h>
#include <OIVShared/ValueSnap.h>

#include <Image.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <string_view>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace OIV
{
    namespace
    {
        // Inspect the IEEE exponent at the input boundary. This stays valid when LTO combines
        // the policy with /fp:fast callers that can otherwise discard floating classification.
        constexpr bool IsFinite(double value)
        {
            static_assert(std::numeric_limits<double>::is_iec559);
            constexpr uint64_t ExponentMask = 0x7ff0000000000000;
            return (std::bit_cast<uint64_t>(value) & ExponentMask) != ExponentMask;
        }
    }  // namespace

    SequencerPolicy::SequencerPolicy(std::function<double()> elapsedSecondsProvider)
        : fMotion(MotionBaseStep, MotionAcceleration, MotionDeceleration, std::move(elapsedSecondsProvider))
    {
    }

    void SequencerPolicy::Reset(std::shared_ptr<IMCodec::Image> image)
    {
        fImage = image != nullptr && image->GetSubImageGroupType() == IMCodec::ImageItemType::AnimationFrame &&
                         image->GetNumSubImages() != 0
                     ? std::move(image)
                     : nullptr;
        fFrameDelays.reset();
        fFrameIntervalMs = 0;
        fMotion.Reset();
        fSpeed = 1.0;
        // With a floor <= 5 ms and a reference speed <= 100%, 100% is always inside the bounds.
        // Only a larger custom floor can require examining the frames during image loading.
        if (fImage != nullptr && fSettings.minFrameIntervalMs > MinSourceDelayMs)
        {
            const auto limits = GetSpeedLimits();
            fSpeed            = std::clamp(fSpeed, limits.minimum, limits.maximum);
        }
    }

    bool SequencerPolicy::Settings::IsValid() const
    {
        // Validate once at the settings boundary. Even the longest uint32_t source delay must
        // have a finite duration at the reference speed, so subsequent divisions cannot overflow.
        constexpr double MinReferenceSpeed = static_cast<double>(std::numeric_limits<uint32_t>::max()) /
                                             std::numeric_limits<double>::max();
        return minFrameIntervalMs > 0 && minFrameIntervalMs <= MaxTimerIntervalMs && slowFrameIntervalMs > 0 &&
               slowFrameIntervalMs <= MaxTimerIntervalMs && IsFinite(slowSpeedPercent) && slowSpeedPercent <= 100.0 &&
               slowSpeedPercent / 100.0 >= MinReferenceSpeed;
    }

    bool SequencerPolicy::SetSettings(const Settings& settings, Clock::time_point now)
    {
        const bool changed = settings.IsValid() && settings != fSettings;
        if (changed)
        {
            fSettings = settings;
            fMotion.Reset();
            if (fImage != nullptr)
            {
                const auto limits = GetSpeedLimits();
                fSpeed            = std::clamp(fSpeed, limits.minimum, limits.maximum);
            }
            RetimeFrame(now);
        }
        return changed;
    }

    SequencerPolicy::SpeedLimits SequencerPolicy::GetSpeedLimits()
    {
        if (!fFrameDelays)
        {
            FrameDelays delays{std::numeric_limits<uint32_t>::max(), MinSourceDelayMs};
            for (uint32_t i = 0; i < fImage->GetNumSubImages(); ++i)
            {
                const auto delay = std::max(
                    MinSourceDelayMs,
                    fImage->GetSubImage(static_cast<uint16_t>(i))->GetAnimationData().delayMilliseconds);
                delays.shortest = std::min(delays.shortest, delay);
                delays.longest  = std::max(delays.longest, delay);
            }
            fFrameDelays = delays;
        }

        // The shortest normalized frame S sets maxSpeed = S / playbackFloor. The longest L
        // sets minSpeed = L / max(slowDuration, L / referenceSpeed), equivalently the LOWER
        // of L / slowDuration and referenceSpeed. Thus default L=20 ms allows 10 s at 0.2%,
        // while L=200 ms allows 20 s at 1%. Averaging delays would miss the limiting frames.
        const double maximum = static_cast<double>(fFrameDelays->shortest) / fSettings.minFrameIntervalMs;
        const double minimum = std::min(static_cast<double>(fFrameDelays->longest) / fSettings.slowFrameIntervalMs,
                                        fSettings.slowSpeedPercent / 100.0);
        // A custom playback floor can conflict with the slow limit. Preserve the floor and
        // collapse the range rather than constructing inverted bounds for std::clamp.
        return {std::min(minimum, maximum), maximum};
    }

    SequencerPolicy::SpeedChange SequencerPolicy::ChangeSpeed(double percent, Clock::time_point now)
    {
        SpeedChange result;
        if (fImage != nullptr && IsFinite(percent) && percent > -100.0 && percent != 0.0)
        {
            const auto limits = GetSpeedLimits();
            // Input timing changes the number of effective percentage steps, not their sign.
            // Raising the positive base to that count preserves isolated +/-1% behavior and
            // avoids negative speed when a long sequence of slowdown commands accelerates.
            const double effectiveSteps = 1.0 + std::abs(fMotion.Add(std::copysign(1.0, percent))) -
                                          InitialMotionVelocity;
            const double requested      = ApplySpeedChange(fSpeed, percent, effectiveSteps);
            const double speed          = std::clamp(requested, limits.minimum, limits.maximum);
            result.changed              = speed != fSpeed;
            fSpeed                      = speed;
            if (result.changed)
                result.retimed = RetimeFrame(now);

            const Limit limit = percent > 0.0 && requested >= limits.maximum   ? Limit::Maximum
                                : percent < 0.0 && requested <= limits.minimum ? Limit::Minimum
                                                                               : Limit::None;
            // Clear accumulated input at either boundary, including blocked repeats. Reversing
            // away from a limit must begin with a fine step instead of carrying old acceleration.
            if (limit != Limit::None)
                fMotion.Reset();
            const double interval = limit == Limit::Minimum
                                        ? FrameIntervalMs(fFrameDelays->longest, fSpeed, fSettings.minFrameIntervalMs)
                                        : fSettings.minFrameIntervalMs;
            result.message        = FormatSpeed(fSpeed, limit, interval);
        }
        return result;
    }

    uint32_t SequencerPolicy::StartFrame(uint32_t delayMilliseconds, Clock::time_point now)
    {
        fFrameDelayMs    = delayMilliseconds;
        fFrameIntervalMs = FrameIntervalMs(delayMilliseconds, fSpeed, fSettings.minFrameIntervalMs);
        fFrameStart      = now;
        fFrameDeadline   = now + std::chrono::milliseconds(fFrameIntervalMs);
        return fFrameIntervalMs;
    }

    bool SequencerPolicy::RetimeFrame(Clock::time_point now)
    {
        const auto previousDeadline = fFrameDeadline;
        if (fFrameIntervalMs != 0)
        {
            const auto interval = FrameIntervalMs(fFrameDelayMs, fSpeed, fSettings.minFrameIntervalMs);
            if (interval != fFrameIntervalMs)
            {
                // The old deadline encodes unplayed progress. Scale only that fraction using
                // the new full duration. An expired frame stays due without another timer reset.
                if (fFrameDeadline > now)
                {
                    const double remainingMs = std::chrono::duration<double, std::milli>(fFrameDeadline - now).count();
                    const auto remaining = std::chrono::duration<double, std::milli>(remainingMs / fFrameIntervalMs *
                                                                                     interval);
                    fFrameDeadline       = now + std::chrono::ceil<Clock::duration>(remaining);
                }
                fFrameIntervalMs = interval;
            }
            // A settings reload can raise the hard floor during a frame. It takes precedence
            // over preserved progress, measured from the frame's start rather than each keypress.
            fFrameDeadline = std::max(fFrameDeadline,
                                      fFrameStart + std::chrono::milliseconds(fSettings.minFrameIntervalMs));
        }
        return fFrameDeadline != previousDeadline;
    }

    uint32_t SequencerPolicy::RemainingFrameMs(Clock::time_point now) const
    {
        uint32_t remaining = 0;
        if (fFrameIntervalMs != 0 && fFrameDeadline > now)
        {
            // Round up to avoid advancing early; zero is reserved for a completed frame.
            remaining = static_cast<uint32_t>(
                std::chrono::ceil<std::chrono::milliseconds>(fFrameDeadline - now).count());
        }
        return remaining;
    }

    bool SequencerPolicy::IsChangeSpeedCommand(const CommandManager::CommandArgs& args)
    {
        return args.GetArgValue("cmd") == "changespeed";
    }

    double SequencerPolicy::ParseSpeedChangePercent(const CommandManager::CommandArgs& args)
    {
        double percent   = 0.0;
        const auto value = args.GetArgValue("amount");
        std::string_view number(value);
        number.remove_prefix(std::min(number.find_first_not_of(" \t\r\n\v\f"), number.size()));
        if (number.starts_with('+'))
            number.remove_prefix(1);

        // from_chars writes through a reference, so classification remains observable under LTO.
        // A floating-return CRT parser (stod/strtod) can inherit /fp:fast return assumptions
        // from other translation units, causing even an explicit infinity check to disappear.
        double parsed           = 0.0;
        const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), parsed);
        if (error == std::errc{} && end == number.data() + number.size() && IsFinite(parsed) && parsed > -100.0)
            percent = parsed;
        return percent;
    }

    double SequencerPolicy::ApplySpeedChange(double currentSpeed, double percent, double effectiveSteps)
    {
        const double requestedSpeed = currentSpeed * std::pow(1.0 + percent / 100.0, effectiveSteps);
        // Match wheel zoom's directional 100% snap without clearing input acceleration. The
        // caller then applies frame-time bounds, which take precedence if 100% is unavailable.
        return SnapToOriginalValue(currentSpeed, requestedSpeed);
    }

    LLUtils::native_string_type SequencerPolicy::FormatSpeed(double speed, Limit limit, double frameIntervalMs)
    {
        LLUtils::native_stringstream number;
        number.imbue(std::locale::classic());
        const double percentage = speed * 100.0;
        int precision           = 2;
        number << std::fixed << std::setprecision(precision) << percentage;
        // Precision belongs to the displayed percentage, not the underlying multiplier. Keep
        // two decimals normally; for a nonzero value rounded to zero, add only enough decimals
        // to show a nonzero digit (0.000456% becomes 0.0005%). Checking actual formatted output
        // also handles rounding ties without a log10 estimate choosing the wrong precision.
        while (percentage != 0.0 &&
               number.view().find_first_of(LLUTILS_TEXT("123456789")) == LLUtils::native_string_type::npos)
        {
            number.str({});
            number << std::setprecision(++precision) << percentage;
        }

        LLUtils::native_stringstream stream;
        stream.imbue(std::locale::classic());
        // Keep the speed readout identical on the first line. A cap adds its explanation below
        // it, so reaching or leaving a limit never replaces or shifts the percentage horizontally.
        stream << LLUTILS_TEXT("<textcolor=#ff8930>Animation speed<textcolor=#7672ff> (") << number.view()
               << LLUTILS_TEXT("%)");
        if (limit == Limit::Maximum)
            stream << LLUTILS_TEXT("\n<textcolor=#ff8930>Maximum animation speed reached<textcolor=#7672ff> "
                                   "(minimum frame time ")
                   << std::fixed << std::setprecision(0) << frameIntervalMs << LLUTILS_TEXT(" ms)");
        else if (limit == Limit::Minimum)
            stream << LLUTILS_TEXT("\n<textcolor=#ff8930>Minimum animation speed reached<textcolor=#7672ff> "
                                   "(maximum frame time ")
                   << std::fixed << std::setprecision(2) << frameIntervalMs / 1000.0 << LLUTILS_TEXT(" s)");
        return stream.str();
    }

    uint32_t SequencerPolicy::NextFrame(uint32_t currentFrame, uint32_t frameCount)
    {
        return frameCount == 0 ? 0 : (currentFrame + 1) % frameCount;
    }

    uint32_t SequencerPolicy::FrameIntervalMs(uint32_t delayMilliseconds, double speed, uint32_t minFrameIntervalMs)
    {
        const double interval = static_cast<double>(std::max(MinSourceDelayMs, delayMilliseconds)) / speed;
        // Clamp before integer conversion: the lower bound covers rounding at maximum speed,
        // and saturation prevents undefined conversion for extreme metadata or slow settings.
        // This requests a timer interval; OS timer resolution and rendering still limit presentation.
        return static_cast<uint32_t>(
            std::clamp(interval, static_cast<double>(minFrameIntervalMs), static_cast<double>(MaxTimerIntervalMs)));
    }
}  // namespace OIV
