package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTokenServer(token string) *Server {
	return &Server{AdminToken: token}
}

func TestRequireAdminToken_NoTokenConfigured(t *testing.T) {
	s := newTokenServer("")
	h := s.requireAdminToken(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("handler should not be reached when token is not configured")
	}))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/admin/reload-model", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("want 503 when ADMIN_TOKEN unset, got %d", rr.Code)
	}
}

func TestRequireAdminToken_MissingHeader(t *testing.T) {
	s := newTokenServer("s3cret")
	h := s.requireAdminToken(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("handler should not be reached without auth header")
	}))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/admin/reload-model", nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", rr.Code)
	}
}

func TestRequireAdminToken_WrongToken(t *testing.T) {
	s := newTokenServer("s3cret")
	h := s.requireAdminToken(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("handler should not be reached with wrong token")
	}))
	req := httptest.NewRequest(http.MethodPost, "/admin/reload-model", nil)
	req.Header.Set("Authorization", "Bearer nope")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", rr.Code)
	}
}

func TestRequireAdminToken_BearerOK(t *testing.T) {
	s := newTokenServer("s3cret")
	called := false
	h := s.requireAdminToken(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodPost, "/admin/reload-model", nil)
	req.Header.Set("Authorization", "Bearer s3cret")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if !called {
		t.Fatal("handler should have been called")
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rr.Code)
	}
}

func TestRequireAdminToken_XAdminTokenOK(t *testing.T) {
	s := newTokenServer("s3cret")
	called := false
	h := s.requireAdminToken(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodPost, "/admin/reload-model", nil)
	req.Header.Set("X-Admin-Token", "s3cret")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if !called {
		t.Fatal("handler should have been called via X-Admin-Token")
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rr.Code)
	}
}
