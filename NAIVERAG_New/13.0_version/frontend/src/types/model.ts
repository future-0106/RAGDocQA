export interface ModelInfo {
  key: string
  type: 'local' | 'api'
  provider: string
  description: string
  is_current: boolean
  params?: Record<string, unknown>
}

export interface ModelState {
  llmModels: ModelInfo[]
  embeddingModels: ModelInfo[]
  currentLlm: string
  currentEmbedding: string
  loading: boolean
}
