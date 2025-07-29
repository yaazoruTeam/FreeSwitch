package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"
)

// RoutingRequest represents the incoming request from FreeSWITCH
type RoutingRequest struct {
	DID string `json:"did"`
	CID string `json:"cid"`
}

// RoutingResponse represents the response to FreeSWITCH
type RoutingResponse struct {
	Provider    string `json:"provider"`
	OutboundDID string `json:"outbound_did"`
	OutboundCID string `json:"outbound_cid"`
	PromptDID   bool   `json:"prompt_did"`
}

// DIDValidationRequest represents a DID validation request
type DIDValidationRequest struct {
	DID string `json:"did"`
}

// DIDValidationResponse represents a DID validation response
type DIDValidationResponse struct {
	Valid    bool   `json:"valid"`
	Message  string `json:"message"`
	Provider string `json:"provider,omitempty"`
}

// CallLogRequest represents a call logging request from FreeSWITCH
type CallLogRequest struct {
	CallID        string `json:"call_id"`
	CallerID      string `json:"caller_id"`
	Destination   string `json:"destination"`
	StartTime     string `json:"start_time"`
	EventType     string `json:"event_type"`
	Provider      string `json:"provider"`
	Status        string `json:"status"`
	Duration      int    `json:"duration"`
	RecordingFile string `json:"recording_file"`
}

// CallLogResponse represents the response to call logging request
type CallLogResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// routeHandler handles incoming routing requests
func routeHandler(w http.ResponseWriter, r *http.Request) {
	// Log the incoming request
	log.Printf("Received routing request from %s", r.RemoteAddr)
	
	// Only accept POST requests
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Parse JSON request
	var req RoutingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Error parsing JSON request: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	log.Printf("Routing request: DID=%s, CID=%s", req.DID, req.CID)
	
	// Apply routing logic
	response := routeCall(req.DID, req.CID)
	
	// Log the response
	log.Printf("Routing response: Provider=%s, OutboundDID=%s, OutboundCID=%s", 
		response.Provider, response.OutboundDID, response.OutboundCID)
	
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	
	// Send JSON response
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding JSON response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}

// routeCall implements the routing logic
func routeCall(did, cid string) RoutingResponse {
	// Default routing response
	response := RoutingResponse{
		Provider:    "provider1_outgoing",
		OutboundDID: did,
		OutboundCID: cid,
	}
	
	// Apply specific routing rules
	switch did {
	case "972792026281":
		// If incoming DID is 972792026281, forward to US number
		response.Provider = "freetelecom_outgoing"
		response.OutboundDID = "18454289532"
		response.OutboundCID = "19296539941"
		log.Printf("Applied route: 972792026281 -> 18454289532 with CID 19296539941")
	
	case "19296539941":
		// If call from 18454289532 to 19296539941, prompt for DID
		if cid == "18454289532" {
			response.PromptDID = true
			response.Provider = "" // No provider yet, need DID input
			response.OutboundDID = ""
			response.OutboundCID = "972792026281"
			log.Printf("Applied prompt rule: 18454289532 -> 19296539941 requires DID input")
		}
	
	default:
		// Use default provider for other numbers
		log.Printf("Applied default routing rule for DID %s", did)
	}
	
	return response
}

// validateDIDHandler validates a DID and returns routing info
func validateDIDHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Received DID validation request from %s", r.RemoteAddr)
	
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req DIDValidationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Error parsing JSON request: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	log.Printf("Validating DID: %s", req.DID)
	
	response := validateDID(req.DID)
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding JSON response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}

// validateDID checks if a DID is valid and returns routing info
func validateDID(did string) DIDValidationResponse {
	// Check if DID matches valid patterns
	if len(did) < 10 {
		return DIDValidationResponse{
			Valid:   false,
			Message: "DID too short, must be at least 10 digits",
		}
	}
	
	// Israeli numbers (972 prefix)
	if len(did) >= 12 && did[:3] == "972" {
		return DIDValidationResponse{
			Valid:    true,
			Message:  "Valid Israeli DID",
			Provider: "freetelecom_outgoing",
		}
	}
	
	// US numbers (1 prefix)
	if len(did) == 11 && did[0] == '1' {
		return DIDValidationResponse{
			Valid:    true,
			Message:  "Valid US DID",
			Provider: "freetelecom_outgoing",
		}
	}
	
	// 10-digit US numbers
	if len(did) == 10 {
		return DIDValidationResponse{
			Valid:    true,
			Message:  "Valid US DID",
			Provider: "freetelecom_outgoing",
		}
	}
	
	return DIDValidationResponse{
		Valid:   false,
		Message: "Invalid DID format",
	}
}

// callLogHandler handles call logging requests from FreeSWITCH
func callLogHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Received call log request from %s", r.RemoteAddr)
	
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req CallLogRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Error parsing call log JSON request: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Log the call data
	log.Printf("Call Log: ID=%s, Caller=%s, Dest=%s, Event=%s, Provider=%s, Status=%s, Duration=%ds, Recording=%s",
		req.CallID, req.CallerID, req.Destination, req.EventType, req.Provider, req.Status, req.Duration, req.RecordingFile)
	
	// Here you could store the call data in a database or file
	// For now, we just log it
	
	response := CallLogResponse{
		Success: true,
		Message: "Call logged successfully",
	}
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding call log response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}

// healthHandler provides a health check endpoint
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func main() {
	// Create HTTP server
	mux := http.NewServeMux()
	
	// Register handlers
	mux.HandleFunc("/route", routeHandler)
	mux.HandleFunc("/validate-did", validateDIDHandler)
	mux.HandleFunc("/call-log", callLogHandler)
	mux.HandleFunc("/health", healthHandler)
	
	// Configure server
	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  30 * time.Second,
	}
	
	log.Printf("Starting HTTP routing server on port 8080")
	log.Printf("Routing endpoint: http://localhost:8080/route")
	log.Printf("DID validation endpoint: http://localhost:8080/validate-did")
	log.Printf("Call logging endpoint: http://localhost:8080/call-log")
	log.Printf("Health check endpoint: http://localhost:8080/health")
	
	// Start server
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}