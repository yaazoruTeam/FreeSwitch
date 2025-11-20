-- HTTP routing script for FreeSWITCH
-- This script calls an HTTP server to determine call routing

local api = freeswitch.API()

-- Configuration
local ROUTING_SERVER_URL = "http://localhost:8080/route"
local TIMEOUT = 3000  -- 3 seconds timeout

-- Function to make HTTP request
function make_http_request(did, cid)
    -- Prepare JSON payload
    local payload = string.format('{"did":"%s","cid":"%s"}', did, cid)
    
    -- Make HTTP POST request
    local curl_cmd = string.format('curl -X POST -H "Content-Type: application/json" -d \'%s\' --connect-timeout 3 --max-time 5 %s', payload, ROUTING_SERVER_URL)
    
    freeswitch.consoleLog("INFO", "HTTP Routing: Making request to " .. ROUTING_SERVER_URL)
    freeswitch.consoleLog("INFO", "HTTP Routing: Payload: " .. payload)
    
    -- Execute curl command
    local response = api:execute("system", curl_cmd)
    
    return response
end

-- Function to parse JSON response
function parse_routing_response(response)
    if not response or response == "" then
        return nil
    end
    
    -- Remove any trailing newlines or spaces
    response = string.gsub(response, "%s+$", "")
    
    freeswitch.consoleLog("INFO", "HTTP Routing: Raw response: " .. response)
    
    -- Simple JSON parsing for our expected format
    local provider = string.match(response, '"provider"%s*:%s*"([^"]+)"')
    local outbound_did = string.match(response, '"outbound_did"%s*:%s*"([^"]+)"')
    local outbound_cid = string.match(response, '"outbound_cid"%s*:%s*"([^"]+)"')
    
    if provider and outbound_did and outbound_cid then
        return {
            provider = provider,
            outbound_did = outbound_did,
            outbound_cid = outbound_cid
        }
    end
    
    return nil
end

-- Function to set default routing
function set_default_routing()
    freeswitch.consoleLog("INFO", "HTTP Routing: Using default routing")
    
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
        freeswitch.consoleLog("ERROR", "HTTP Routing: Missing DID or CID information")
        set_default_routing()
        return
    end
    
    freeswitch.consoleLog("INFO", "HTTP Routing: Processing call - DID: " .. incoming_did .. ", CID: " .. incoming_cid)
    
    -- Make HTTP request
    local response = make_http_request(incoming_did, incoming_cid)
    
    if not response then
        freeswitch.consoleLog("ERROR", "HTTP Routing: No response from server")
        set_default_routing()
        return
    end
    
    -- Parse response
    local routing_info = parse_routing_response(response)
    
    if not routing_info then
        freeswitch.consoleLog("ERROR", "HTTP Routing: Invalid response format")
        set_default_routing()
        return
    end
    
    -- Set routing variables
    session:setVariable("routing_provider", routing_info.provider)
    session:setVariable("routing_did", routing_info.outbound_did)
    session:setVariable("routing_cid", routing_info.outbound_cid)
    
    freeswitch.consoleLog("INFO", "HTTP Routing: Set provider=" .. routing_info.provider .. 
                          ", outbound_did=" .. routing_info.outbound_did .. 
                          ", outbound_cid=" .. routing_info.outbound_cid)
end

-- Execute routing if session exists
if session then
    route_call()
else
    freeswitch.consoleLog("ERROR", "HTTP Routing: No session available")
end