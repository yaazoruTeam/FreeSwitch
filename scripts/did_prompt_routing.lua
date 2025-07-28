-- DID Prompt routing script for FreeSWITCH
-- This script handles DID prompting when required by routing server

local api = freeswitch.API()

-- Configuration
local ROUTING_SERVER_URL = "http://localhost:8080/route"
local DID_VALIDATION_URL = "http://localhost:8080/validate-did"
local TIMEOUT = 3000  -- 3 seconds timeout

-- Function to make HTTP request for initial routing
function make_routing_request(did, cid)
    local payload = string.format('{"did":"%s","cid":"%s"}', did, cid)
    local curl_cmd = string.format('curl -X POST -H "Content-Type: application/json" -d \'%s\' --connect-timeout 3 --max-time 5 %s', payload, ROUTING_SERVER_URL)
    
    freeswitch.consoleLog("INFO", "DID Prompt: Making routing request to " .. ROUTING_SERVER_URL)
    freeswitch.consoleLog("INFO", "DID Prompt: Payload: " .. payload)
    
    local response = api:execute("system", curl_cmd)
    return response
end

-- Function to validate a DID
function validate_did(did)
    local payload = string.format('{"did":"%s"}', did)
    local curl_cmd = string.format('curl -X POST -H "Content-Type: application/json" -d \'%s\' --connect-timeout 3 --max-time 5 %s', payload, DID_VALIDATION_URL)
    
    freeswitch.consoleLog("INFO", "DID Prompt: Validating DID: " .. did)
    
    local response = api:execute("system", curl_cmd)
    return response
end

-- Function to parse routing response
function parse_routing_response(response)
    if not response or response == "" then
        return nil
    end
    
    response = string.gsub(response, "%s+$", "")
    freeswitch.consoleLog("INFO", "DID Prompt: Routing response: " .. response)
    
    local provider = string.match(response, '"provider"%s*:%s*"([^"]*)"')
    local outbound_did = string.match(response, '"outbound_did"%s*:%s*"([^"]*)"')
    local outbound_cid = string.match(response, '"outbound_cid"%s*:%s*"([^"]+)"')
    local prompt_did = string.match(response, '"prompt_did"%s*:%s*([^,}]+)')
    
    return {
        provider = provider,
        outbound_did = outbound_did,
        outbound_cid = outbound_cid,
        prompt_did = (prompt_did == "true")
    }
end

-- Function to parse DID validation response
function parse_validation_response(response)
    if not response or response == "" then
        return nil
    end
    
    response = string.gsub(response, "%s+$", "")
    freeswitch.consoleLog("INFO", "DID Prompt: Validation response: " .. response)
    
    local valid = string.match(response, '"valid"%s*:%s*([^,}]+)')
    local provider = string.match(response, '"provider"%s*:%s*"([^"]*)"')
    local message = string.match(response, '"message"%s*:%s*"([^"]+)"')
    
    return {
        valid = (valid == "true"),
        provider = provider,
        message = message
    }
end

-- Function to prompt for DID
function prompt_for_did(routing_info)
    freeswitch.consoleLog("INFO", "DID Prompt: Prompting user for DID")
    
    -- Answer the call to start IVR
    session:answer()
    session:sleep(500)
    
    local max_attempts = 3
    local attempt = 1
    
    while attempt <= max_attempts do
        freeswitch.consoleLog("INFO", "DID Prompt: Attempt " .. attempt .. " of " .. max_attempts)
        
        -- Play prompt message
        session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-please_enter_the_phone_number.wav")
        session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-followed_by_pound.wav")
        
        -- Get digits from user
        local digits = session:playAndGetDigits(10, 15, 3, 5000, "#", 
            "/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-please_enter_the_phone_number.wav",
            "/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-that_was_an_invalid_entry.wav",
            "\\d+")
        
        if digits and digits ~= "" then
            freeswitch.consoleLog("INFO", "DID Prompt: User entered: " .. digits)
            
            -- Validate the DID
            local validation_response = validate_did(digits)
            local validation_info = parse_validation_response(validation_response)
            
            if validation_info and validation_info.valid then
                freeswitch.consoleLog("INFO", "DID Prompt: DID validated successfully")
                session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-thank_you.wav")
                
                -- Set routing variables for the validated DID
                session:setVariable("routing_provider", validation_info.provider or "freetelecom_outgoing")
                session:setVariable("routing_did", digits)
                session:setVariable("routing_cid", routing_info.outbound_cid)
                session:setVariable("did_prompt_complete", "true")
                
                return true
            else
                freeswitch.consoleLog("INFO", "DID Prompt: Invalid DID entered")
                session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-that_was_an_invalid_entry.wav")
                attempt = attempt + 1
            end
        else
            freeswitch.consoleLog("INFO", "DID Prompt: No digits entered")
            session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-no_input_detected.wav")
            attempt = attempt + 1
        end
    end
    
    -- Max attempts reached
    freeswitch.consoleLog("INFO", "DID Prompt: Max attempts reached, hanging up")
    session:streamFile("/usr/share/freeswitch/sounds/en/us/callie/ivr/8000/ivr-goodbye.wav")
    session:hangup()
    return false
end

-- Function to set default routing
function set_default_routing()
    freeswitch.consoleLog("INFO", "DID Prompt: Using default routing")
    
    session:setVariable("routing_provider", "provider1_outgoing")
    session:setVariable("routing_did", session:getVariable("destination_number"))
    session:setVariable("routing_cid", session:getVariable("caller_id_number"))
end

-- Main routing logic
function route_call()
    -- Get incoming call information
    local incoming_did = session:getVariable("destination_number")
    local incoming_cid = session:getVariable("caller_id_number")
    
    if not incoming_did or not incoming_cid then
        freeswitch.consoleLog("ERROR", "DID Prompt: Missing DID or CID information")
        set_default_routing()
        return
    end
    
    freeswitch.consoleLog("INFO", "DID Prompt: Processing call - DID: " .. incoming_did .. ", CID: " .. incoming_cid)
    
    -- Make initial routing request
    local response = make_routing_request(incoming_did, incoming_cid)
    
    if not response then
        freeswitch.consoleLog("ERROR", "DID Prompt: No response from server")
        set_default_routing()
        return
    end
    
    -- Parse response
    local routing_info = parse_routing_response(response)
    
    if not routing_info then
        freeswitch.consoleLog("ERROR", "DID Prompt: Invalid response format")
        set_default_routing()
        return
    end
    
    -- Check if DID prompt is required
    if routing_info.prompt_did then
        freeswitch.consoleLog("INFO", "DID Prompt: DID prompt required")
        
        if not prompt_for_did(routing_info) then
            -- Prompt failed, call will be hung up
            return
        end
        
        -- DID prompt successful, routing variables already set
        freeswitch.consoleLog("INFO", "DID Prompt: DID prompt completed successfully")
        
    else
        -- Normal routing
        session:setVariable("routing_provider", routing_info.provider)
        session:setVariable("routing_did", routing_info.outbound_did)
        session:setVariable("routing_cid", routing_info.outbound_cid)
        
        freeswitch.consoleLog("INFO", "DID Prompt: Set provider=" .. routing_info.provider .. 
                              ", outbound_did=" .. routing_info.outbound_did .. 
                              ", outbound_cid=" .. routing_info.outbound_cid)
    end
end

-- Execute routing if session exists
if session then
    route_call()
else
    freeswitch.consoleLog("ERROR", "DID Prompt: No session available")
end