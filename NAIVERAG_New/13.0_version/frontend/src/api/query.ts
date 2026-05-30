import request from './request'
import type { QueryRequest, QueryResponse } from '@/types/api'

export function sendQuery(params: QueryRequest): Promise<QueryResponse> {
  return request.post('/query', params)
}
