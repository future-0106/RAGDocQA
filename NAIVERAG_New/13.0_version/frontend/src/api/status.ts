import request from './request'
import type { StatusResponse, HealthResponse, RetrievalConfigResponse } from '@/types/api'

export function getStatus(): Promise<StatusResponse> {
  return request.get('/status')
}

export function getHealth(): Promise<HealthResponse> {
  return request.get('/health')
}

export function getRetrievalConfig(): Promise<RetrievalConfigResponse> {
  return request.get('/retrieval-config')
}

export function updateRetrievalConfig(config: Record<string, unknown>): Promise<Record<string, unknown>> {
  return request.post('/update-retrieval-config', config)
}
