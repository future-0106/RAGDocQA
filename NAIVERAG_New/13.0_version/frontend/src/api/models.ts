import request from './request'
import type { ModelsResponse, HealthResponse } from '@/types/api'

export function getModels(): Promise<ModelsResponse> {
  return request.get('/models')
}

export function switchModel(model_key: string): Promise<Record<string, unknown>> {
  return request.post('/switch-model', { model_key })
}

export function switchEmbedding(model_key: string): Promise<Record<string, unknown>> {
  return request.post('/switch-embedding', { model_key })
}
