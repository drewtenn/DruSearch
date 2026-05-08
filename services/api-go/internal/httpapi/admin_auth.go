package httpapi

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// requireAdminToken gates /admin/* on a shared bearer token.
//
// If AdminToken is unset the server refuses admin requests outright — the
// production posture is "no token configured = no admin access," not "open."
// To exercise admin routes locally, set ADMIN_TOKEN in the environment.
func (s *Server) requireAdminToken(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.AdminToken == "" {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{
				"error": "admin disabled: ADMIN_TOKEN not configured",
			})
			return
		}
		if !validAdminToken(r, s.AdminToken) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="drusearch-admin"`)
			writeJSON(w, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func validAdminToken(r *http.Request, expected string) bool {
	if h := r.Header.Get("Authorization"); h != "" {
		if rest, ok := strings.CutPrefix(h, "Bearer "); ok {
			return constantTimeEqual(rest, expected)
		}
	}
	if h := r.Header.Get("X-Admin-Token"); h != "" {
		return constantTimeEqual(h, expected)
	}
	return false
}

func constantTimeEqual(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}
