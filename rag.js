/**
 * rag.js — In-browser TF-IDF retriever for webSLMDemo
 * No dependencies. Works entirely in the browser with a pre-loaded document array.
 */

export class TFIDFRetriever {
  constructor(documents) {
    this.documents = documents;
    this._idf = {};
    this._tfVectors = [];
    this._buildIndex();
  }

  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 2);
  }

  _buildIndex() {
    const N = this.documents.length;
    if (N === 0) return;

    const df = {};
    const tokensList = this.documents.map(doc =>
      this._tokenize(doc.title + ' ' + doc.content)
    );

    // Document frequency
    tokensList.forEach(tokens => {
      new Set(tokens).forEach(t => {
        df[t] = (df[t] || 0) + 1;
      });
    });

    // IDF with smoothing
    Object.keys(df).forEach(t => {
      this._idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1;
    });

    // TF-IDF vectors
    this._tfVectors = tokensList.map(tokens => {
      const tf = {};
      tokens.forEach(t => {
        tf[t] = (tf[t] || 0) + 1 / tokens.length;
      });
      const vec = {};
      Object.entries(tf).forEach(([t, v]) => {
        vec[t] = v * (this._idf[t] || 0);
      });
      return vec;
    });
  }

  retrieve(query, k = 3) {
    if (this.documents.length === 0) return [];

    const qTokens = this._tokenize(query);
    if (qTokens.length === 0) return [];

    const qVec = {};
    qTokens.forEach(t => {
      qVec[t] = (qVec[t] || 0) + 1;
    });

    const qNorm = Math.sqrt(
      Object.values(qVec).reduce((s, v) => s + v * v, 0)
    );

    const scores = this._tfVectors.map((dVec, i) => {
      let dot = 0;
      Object.entries(qVec).forEach(([t, qv]) => {
        dot += qv * (dVec[t] || 0);
      });
      const dNorm = Math.sqrt(
        Object.values(dVec).reduce((s, v) => s + v * v, 0)
      );
      const score = qNorm > 0 && dNorm > 0 ? dot / (qNorm * dNorm) : 0;
      return { doc: this.documents[i], score };
    });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .filter(r => r.score > 0.01);
  }
}
