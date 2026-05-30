export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: SourceItemData[]
  metadata?: {
    processing_time?: number
    model_used?: string
    retrieval_mode?: string
    source_count?: number
  }
}

export interface SourceItemData {
  content: string
  score: number
  rank: number
}

export interface ChatState {
  messages: ChatMessage[]
  isProcessing: boolean
  k: number
  enableRewriting: boolean | null
}
