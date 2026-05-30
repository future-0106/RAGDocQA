export interface ApiResponse<T = unknown> {
  success: boolean
  error?: string
  data?: T
}

export interface QueryRequest {
  question: string
  k?: number
  score_threshold?: number
  include_sources?: boolean
  enable_rewriting?: boolean | null
}

export interface SourceItem {
  content: string
  score: number
  rank: number
}

export interface QueryResponse {
  success: boolean
  question: string
  answer: string
  model_used: string
  processing_time: number
  sources: SourceItem[]
  context_length: number
  source_count: number
  retrieval_mode: string
  reranker_enabled: boolean
  hybrid_weights?: [number, number]
  rewritten_queries?: string[]
  timestamp: string
}

export interface ModelItem {
  key: string
  type: string
  provider: string
  description: string
  is_current: boolean
  params?: Record<string, unknown>
}

export interface ModelsResponse {
  success: boolean
  llm_models: ModelItem[]
  embedding_models: ModelItem[]
  current_llm: string
  current_embedding: string
  timestamp: string
}

export interface FileInfo {
  name: string
  size: number
  modified: string
  type: string
}

export interface FilesResponse {
  success: boolean
  count: number
  files: string[]
  details: FileInfo[]
  timestamp: string
}

export interface UploadResponse {
  success: boolean
  message: string
  filename: string
  file_size: number
  document_count: number
  processing_time: number
}

export interface SystemStatus {
  status: string
  timestamp: string
  models: {
    llm: string
    llm_info: Record<string, unknown>
    embedding: string
    embedding_info: Record<string, unknown>
    rewrite_llm: string
    rewrite_enabled: boolean
  }
  device: string
  config: {
    chunk_size: number
    chunk_overlap: number
    similarity_top_k: number
    score_threshold: number
    max_context_length: number
  }
  vector_store: {
    collection_name: string
    document_count: number
    persist_directory: string
    status: string
  }
  files: {
    count: number
    list: string[]
    details: FileInfo[]
  }
  history: {
    count: number
    max: number
  }
}

export interface StatusResponse {
  success: boolean
  data: SystemStatus
}

export interface HistoryItem {
  question: string
  timestamp: string
  model: string
}

export interface HistoryResponse {
  success: boolean
  count: number
  total: number
  max: number
  history: HistoryItem[]
  timestamp: string
}

export interface RetrievalConfig {
  retrieval_mode: string
  hybrid_weights: [number, number]
  reranker_enabled: boolean
  reranker_top_k: number
  total_documents: number
  has_bm25_index: boolean
  has_reranker: boolean
}

export interface RetrievalConfigResponse {
  success: boolean
  config: RetrievalConfig
  timestamp: string
}

export interface ProcessDocumentsResponse {
  success: boolean
  file_count: number
  document_count: number
  skipped_count: number
  skipped_files?: string[]
  processing_time: number
  timestamp: string
}

export interface HealthResponse {
  status: string
  timestamp: string
  system: string
  version: string
  components: Record<string, string>
}
