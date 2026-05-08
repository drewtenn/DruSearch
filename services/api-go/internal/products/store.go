package products

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Product struct {
	ProductID    string          `json:"product_id"`
	Title        string          `json:"title"`
	Description  string          `json:"description"`
	BulletPoints string          `json:"bullet_points"`
	Brand        string          `json:"brand"`
	Color        string          `json:"color"`
	Category     string          `json:"category"`
	CategoryPath []string        `json:"category_path"`
	Locale       string          `json:"locale"`
	PriceCents   int             `json:"price_cents"`
	RawMetadata  json.RawMessage `json:"raw_metadata"`
}

type Store struct{ pool *pgxpool.Pool }

func New(pool *pgxpool.Pool) *Store { return &Store{pool: pool} }

func (s *Store) ByID(ctx context.Context, id string) (*Product, error) {
	const q = `
        SELECT product_id, title,
               COALESCE(description, ''),
               COALESCE(bullet_points, ''),
	               COALESCE(brand, ''),
	               COALESCE(color, ''),
	               COALESCE(category, ''),
	               category_path,
	               locale,
	               COALESCE(price_cents, 0),
	               COALESCE(raw_metadata, '{}'::jsonb)
	        FROM products
	        WHERE product_id = $1`
	var p Product
	err := s.pool.QueryRow(ctx, q, id).Scan(
		&p.ProductID, &p.Title, &p.Description, &p.BulletPoints,
		&p.Brand, &p.Color, &p.Category, &p.CategoryPath, &p.Locale, &p.PriceCents,
		&p.RawMetadata,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}
