-- Table des prédictions (inputs et outputs du modèle)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_data JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    probability NUMERIC(5,4) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
