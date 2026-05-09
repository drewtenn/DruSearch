package httpapi

import "testing"

func TestRankerFromRequestDefaultsAndAliases(t *testing.T) {
	tests := []struct {
		name    string
		query   string
		def     string
		want    rankerMode
		wantErr bool
	}{
		{name: "default hybrid", def: "", want: rankerHybrid},
		{name: "default bge rejected", def: "bge", wantErr: true},
		{name: "query overrides invalid default", query: "hybrid", def: "bge", want: rankerHybrid},
		{name: "rrf alias", query: "rrf", want: rankerHybrid},
		{name: "ltr", query: "ltr", want: rankerLTR},
		{name: "bge query rejected", query: "bge", wantErr: true},
		{name: "invalid", query: "wat", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := rankerFromRequest(tt.query, tt.def)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("rankerFromRequest: %v", err)
			}
			if got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}
