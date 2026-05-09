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
		{name: "default bge", def: "bge", want: rankerBGE},
		{name: "query overrides default", query: "hybrid", def: "bge", want: rankerHybrid},
		{name: "rrf alias", query: "rrf", want: rankerHybrid},
		{name: "ltr", query: "ltr", want: rankerLTR},
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
