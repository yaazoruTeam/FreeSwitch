-- Call Logger script for FreeSWITCH
-- Captures call information and sends to HTTP endpoint for billing/recording

local api = freeswitch.API()

-- Configuration
local CALL_LOG_URL = "http://localhost:8080/call-log"
local TIMEOUT = 3000  -- 3 seconds timeout

-- Function to make HTTP request to log call data
function log_call_data(call_data)
    local payload = string.format('{"call_id":"%s","caller_id":"%s","destination":"%s","start_time":"%s","event_type":"%s","provider":"%s","status":"%s","duration":%d,"recording_file":"%s"}',
        call_data.call_id or "",
        call_data.caller_id or "",
        call_data.destination or "",
        call_data.start_time or "",
        call_data.event_type or "",
        call_data.provider or "",
        call_data.status or "",
        call_data.duration or 0,
        call_data.recording_file or ""
    )
    
    local curl_cmd = string.format('curl -X POST -H "Content-Type: application/json" -d \'%s\' --connect-timeout 3 --max-time 5 %s', payload, CALL_LOG_URL)
    
    freeswitch.consoleLog("INFO", "Call Logger: Sending call data to " .. CALL_LOG_URL)
    freeswitch.consoleLog("DEBUG", "Call Logger: Payload: " .. payload)
    
    local response = api:execute("system", curl_cmd)
    
    if response and response ~= "" then
        freeswitch.consoleLog("INFO", "Call Logger: HTTP response: " .. response)
    else
        freeswitch.consoleLog("WARNING", "Call Logger: No response from logging endpoint")
    end
end

-- Function to get current timestamp in ISO format
function get_timestamp()
    return os.date("!%Y-%m-%dT%H:%M:%SZ")
end

-- Function to log call start event
function log_call_start()
    if not session then
        freeswitch.consoleLog("ERROR", "Call Logger: No session available")
        return
    end
    
    local call_data = {
        call_id = session:getVariable("uuid") or session:get_uuid(),
        caller_id = session:getVariable("caller_id_number") or "",
        destination = session:getVariable("destination_number") or "",
        start_time = get_timestamp(),
        event_type = "call_start",
        provider = session:getVariable("routing_provider") or "unknown",
        status = "started",
        duration = 0,
        recording_file = session:getVariable("recording_filename") or ""
    }
    
    freeswitch.consoleLog("INFO", "Call Logger: Logging call start - UUID: " .. call_data.call_id)
    log_call_data(call_data)
    
    -- Store start time for duration calculation
    session:setVariable("call_start_time", os.time())
end

-- Function to log call answer event
function log_call_answer()
    if not session then
        return
    end
    
    local call_data = {
        call_id = session:getVariable("uuid") or session:get_uuid(),
        caller_id = session:getVariable("caller_id_number") or "",
        destination = session:getVariable("destination_number") or "",
        start_time = get_timestamp(),
        event_type = "call_answer",
        provider = session:getVariable("routing_provider") or "unknown",
        status = "answered",
        duration = 0,
        recording_file = session:getVariable("recording_filename") or ""
    }
    
    freeswitch.consoleLog("INFO", "Call Logger: Logging call answer - UUID: " .. call_data.call_id)
    log_call_data(call_data)
    
    -- Store answer time
    session:setVariable("call_answer_time", os.time())
end

-- Function to log call end event
function log_call_end()
    if not session then
        return
    end
    
    local start_time = session:getVariable("call_start_time")
    local duration = 0
    
    if start_time then
        duration = os.time() - tonumber(start_time)
    end
    
    local hangup_cause = session:getVariable("hangup_cause") or "UNKNOWN"
    local call_status = "completed"
    
    -- Determine call status based on hangup cause
    if hangup_cause == "NORMAL_CLEARING" then
        call_status = "completed"
    elseif hangup_cause == "USER_BUSY" then
        call_status = "busy"
    elseif hangup_cause == "NO_ANSWER" then
        call_status = "no_answer"
    elseif hangup_cause == "CALL_REJECTED" then
        call_status = "rejected"
    else
        call_status = "failed"
    end
    
    local call_data = {
        call_id = session:getVariable("uuid") or session:get_uuid(),
        caller_id = session:getVariable("caller_id_number") or "",
        destination = session:getVariable("destination_number") or "",
        start_time = get_timestamp(),
        event_type = "call_end",
        provider = session:getVariable("routing_provider") or "unknown",
        status = call_status,
        duration = duration,
        recording_file = session:getVariable("recording_filename") or ""
    }
    
    freeswitch.consoleLog("INFO", "Call Logger: Logging call end - UUID: " .. call_data.call_id .. ", Duration: " .. duration .. "s, Status: " .. call_status)
    log_call_data(call_data)
end

-- Function to log bridge attempt
function log_bridge_attempt()
    if not session then
        return
    end
    
    local call_data = {
        call_id = session:getVariable("uuid") or session:get_uuid(),
        caller_id = session:getVariable("caller_id_number") or "",
        destination = session:getVariable("routing_did") or session:getVariable("destination_number") or "",
        start_time = get_timestamp(),
        event_type = "bridge_attempt",
        provider = session:getVariable("routing_provider") or "unknown",
        status = "bridging",
        duration = 0,
        recording_file = session:getVariable("recording_filename") or ""
    }
    
    freeswitch.consoleLog("INFO", "Call Logger: Logging bridge attempt - UUID: " .. call_data.call_id)
    log_call_data(call_data)
end

-- Main function to determine which event to log
function log_call_event(event_type)
    if not session then
        freeswitch.consoleLog("ERROR", "Call Logger: No session available")
        return
    end
    
    event_type = event_type or "call_start"
    
    freeswitch.consoleLog("INFO", "Call Logger: Processing event: " .. event_type)
    
    if event_type == "call_start" then
        log_call_start()
    elseif event_type == "call_answer" then
        log_call_answer()
    elseif event_type == "call_end" then
        log_call_end()
    elseif event_type == "bridge_attempt" then
        log_bridge_attempt()
    else
        freeswitch.consoleLog("WARNING", "Call Logger: Unknown event type: " .. event_type)
    end
end

-- If script is called with an argument, use it as event type
local event_type = argv and argv[1] or "call_start"
log_call_event(event_type)