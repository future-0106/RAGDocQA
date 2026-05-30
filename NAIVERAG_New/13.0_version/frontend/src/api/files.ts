import request from './request'
import type { FilesResponse, UploadResponse, ProcessDocumentsResponse } from '@/types/api'

export function getFiles(): Promise<FilesResponse> {
  return request.get('/files')
}

export function deleteFile(filename: string): Promise<{ success: boolean }> {
  return request.delete(`/files/${encodeURIComponent(filename)}`)
}

export function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return request.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
}

export function processDirectory(source_dir: string): Promise<ProcessDocumentsResponse> {
  return request.post('/data/process-documents', { source_dir })
}
