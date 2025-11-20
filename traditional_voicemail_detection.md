# Traditional Voicemail Detection Methods

## Overview

Traditional voicemail detection in telephony systems relies on audio signal processing and call flow analysis to determine when a call has reached an answering machine or voicemail system. This document covers the primary methods used in FreeSWITCH and other telephony platforms.

## FreeSWITCH Detection Modules

### 1. mod_avmd (Advanced Voice Mail Detection)

**Algorithm**: DESA-2 (Discrete Energy Separation Algorithm)
- Designed to detect single-frequency constant-amplitude tones
- Primary use: detecting "beep" sounds at the end of voicemail/answering machine greetings
- Fast, computationally simple but "a little inaccurate in presence of noise"

**Technical Details**:
```xml
<!-- AVMD Configuration -->
<configuration name="avmd.conf" description="AVMD Configuration">
  <settings>
    <param name="sample_n_continuous_streak" value="3"/>
    <param name="sample_n_to_skip" value="0"/>
    <param name="require_continuous_streak" value="true"/>
    <param name="sample_n_continuous_streak_multiplier" value="1.0"/>
  </settings>
</configuration>
```

**Performance Impact**:
- Tests on 8-core Intel i7-4790K CPU show 10x higher CPU usage compared to simple playback
- Default configuration uses 36 detectors

**Usage in Dialplan**:
```xml
<action application="avmd_start" data="inbound_channel=1,outbound_channel=0"/>
```

**Event Handling**:
- Fires `avmd::beep` event when beep detected
- Sets channel variable `avmd_detect=true`
- Delivers events to ESL socket listeners and logs

### 2. mod_vmd (Voice Mail Detection) - DEPRECATED

**Status**: Deprecated and will be removed in future FreeSWITCH versions
**Recommendation**: Use mod_avmd instead

**Historical Context**:
- Older implementation with limited accuracy
- Replaced by more advanced mod_avmd

### 3. mod_com_amd (Commercial Answering Machine Detection)

**Cost**: $50 per channel license
**Features**:
- Sophisticated audio analysis algorithms
- Word counting for machine vs human detection
- Higher accuracy than free alternatives

**Configuration Parameters**:
```xml
<param name="MachineWordsThreshold" value="6"/>
```
- Default: 6 words before classifying as machine
- Logic: Machine greetings typically use more words than human greetings
- Sets `amd_status` variable to "human" or "machine"

**Accuracy**: No algorithm achieves 100% accuracy, but commercial solutions typically perform better

## SIP Response Code Analysis

### Common SIP Response Codes for Voicemail Detection

| Code | Description | Voicemail Indicator |
|------|-------------|-------------------|
| 486 | Busy Here | Possible VM routing |
| 487 | Request Terminated | Early hangup pattern |
| 480 | Temporarily Unavailable | Default for unspecified causes |

### Hangup Cause Analysis

FreeSWITCH provides detailed hangup cause codes that can indicate voicemail scenarios:

**Key Variables**:
- `bridge_hangup_cause`: For external targets
- `originate_disposition`: For internal SIP phones
- `hangup_cause`: General call termination reason

**Common Hangup Causes**:
```lua
-- Lua example for hangup cause analysis
local hangup_cause = session:getVariable("hangup_cause")
if hangup_cause == "NO_ANSWER" then
    -- Likely voicemail after timeout
elseif hangup_cause == "USER_BUSY" then
    -- May redirect to voicemail
elseif hangup_cause == "CALL_REJECTED" then
    -- Possible immediate VM transfer
end
```

## Call Duration Pattern Analysis

### Short Call Detection
- Calls answered and immediately terminated (< 5 seconds)
- Often indicates automated voicemail pickup
- Combined with other indicators for better accuracy

### Timeout-Based Detection
```xml
<!-- Dialplan timeout configuration -->
<action application="set" data="call_timeout=30"/>
<action application="set" data="hangup_after_bridge=true"/>
```

## Implementation Challenges

### Accuracy Limitations
- **User Report**: "accuracy is so poor. I would even say that it's useless"
- **Signal Quality Impact**: Poor audio quality increases false positives
- **Noise Sensitivity**: Background noise affects beep detection
- **Regional Variations**: Different voicemail systems use different beep frequencies

### False Positives/Negatives
- **False Positives**: Music or tones mistaken for beeps
- **False Negatives**: Missed beeps due to frequency variations
- **Custom Messages**: Personalized voicemail messages without standard beeps

## Best Practices

### 1. Multi-Method Approach
```lua
-- Combine multiple detection methods
function detect_voicemail()
    local avmd_result = session:getVariable("avmd_detect")
    local call_duration = get_call_duration()
    local hangup_cause = session:getVariable("hangup_cause")
    
    if avmd_result == "true" or 
       (call_duration < 5 and hangup_cause == "NO_ANSWER") then
        return true
    end
    return false
end
```

### 2. Timeout Configuration
- Set appropriate call timeouts based on expected answer patterns
- Use `continue_on_fail=true` for proper call flow handling
- Configure `bridge_early_media=true` for better audio analysis

### 3. Provider-Specific Logic
Different carriers and regions have varying voicemail behaviors:
```lua
-- Provider-specific detection logic
local provider = session:getVariable("routing_provider")
if provider == "freetelecom_outgoing" then
    -- Apply specific timeout and detection parameters
    session:setVariable("call_timeout", "25")
end
```

## Performance Considerations

### CPU Usage
- mod_avmd: 10x CPU usage increase
- Recommended for systems with adequate processing power
- Monitor system performance under load

### Memory Usage
- Detection threads consume additional memory
- Scale based on concurrent call volume
- Consider system limits when configuring detectors

## Configuration Examples

### Basic AVMD Setup
```xml
<!-- modules.conf.xml -->
<load module="mod_avmd"/>

<!-- dialplan example -->
<extension name="voicemail_detection">
  <condition field="destination_number" expression="^(.*)$">
    <action application="avmd_start" data="inbound_channel=1"/>
    <action application="bridge" data="sofia/gateway/provider/${destination_number}"/>
    <action application="lua" data="handle_avmd_result.lua"/>
  </condition>
</extension>
```

### Event Handling Script
```lua
-- handle_avmd_result.lua
local avmd_detected = session:getVariable("avmd_detect")
if avmd_detected == "true" then
    freeswitch.consoleLog("INFO", "Voicemail detected via AVMD")
    -- Redirect to custom voicemail handling
    session:execute("transfer", "voicemail_handler XML default")
else
    freeswitch.consoleLog("INFO", "Live answer detected")
end
```

## Troubleshooting

### Common Issues
1. **No AVMD Events**: Check module loading and dialplan configuration
2. **High False Positives**: Adjust detection sensitivity parameters
3. **Performance Issues**: Monitor CPU usage and consider reducing detector count
4. **Inconsistent Results**: Verify audio quality and network conditions

### Debugging Commands
```bash
# Check module status
fs_cli -x "module_exists mod_avmd"

# Monitor AVMD events
fs_cli -x "events plain CUSTOM avmd::beep"

# Check channel variables
fs_cli -x "uuid_dump <uuid>"
```

## Conclusion

Traditional voicemail detection methods in FreeSWITCH provide a foundation for call routing decisions but have inherent accuracy limitations. While mod_avmd offers the best free solution using beep detection, real-world accuracy challenges necessitate combining multiple detection methods and considering upgrade paths to AI-based solutions for production environments requiring higher reliability.

The key to successful traditional voicemail detection lies in:
1. Combining multiple detection methods
2. Provider-specific configuration optimization
3. Appropriate timeout and threshold settings
4. Continuous monitoring and adjustment based on actual call patterns

For critical applications where accuracy is paramount, consider commercial solutions or modern AI-based detection systems that can achieve significantly higher accuracy rates.