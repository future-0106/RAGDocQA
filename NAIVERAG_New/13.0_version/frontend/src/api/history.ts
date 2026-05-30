import request from './request'
import type { HistoryResponse } from '@/types/api'

export function getHistory(limit = 50): Promise<HistoryResponse> {
  return request.get(`/history?limit=${limit}`)
}

export function clearHistory(): Promise<{ success: boolean }> {
  return request.delete('/history')
}
