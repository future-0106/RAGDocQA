<script setup lang="ts">
import { ref } from 'vue'
import { uploadFile, processDirectory } from '@/api/files'

const uploading = ref(false)
const processing = ref(false)
const result = ref('')
const directoryPath = ref('')
const fileInput = ref<HTMLInputElement | null>(null)

async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  uploading.value = true
  result.value = ''
  try {
    const data = await uploadFile(file)
    if (data.success) {
      result.value = `✅ 上传成功！文档数量: ${data.document_count}`
    } else {
      result.value = `❌ 上传失败: ${data.message}`
    }
  } catch (e: unknown) {
    result.value = `❌ 上传失败: ${e instanceof Error ? e.message : '未知错误'}`
  } finally {
    uploading.value = false
  }
}

async function processDir() {
  const dir = directoryPath.value.trim()
  if (!dir) return

  processing.value = true
  result.value = ''
  try {
    const data = await processDirectory(dir)
    if (data.success) {
      let html = `✅ 处理完成！新增文件: ${data.file_count}，文档块: ${data.document_count}`
      if (data.skipped_count > 0) {
        html += `<br><span style="color:#fbbf24">跳过重复: ${data.skipped_count} 个</span>`
      }
      html += `<br>耗时: ${data.processing_time?.toFixed(2)}秒`
      result.value = html
    } else {
      result.value = `❌ 处理失败`
    }
  } catch (e: unknown) {
    result.value = `❌ 请求失败: ${e instanceof Error ? e.message : '未知错误'}`
  } finally {
    processing.value = false
  }
}
</script>

<template>
  <div>
    <h2 class="text-2xl font-semibold text-white mb-6">
      <i class="fas fa-cloud-upload-alt text-neon-cyan mr-3"></i>文件上传
    </h2>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="card p-6">
        <h3 class="font-semibold text-lg mb-4 text-white">单文件上传</h3>
        <div class="upload-zone" @click="fileInput?.click()">
          <i class="fas fa-cloud-upload-alt text-5xl text-slate-600 mb-4"></i>
          <p class="text-slate-300 mb-2">点击选择文件</p>
          <p class="text-sm text-slate-500">支持 PDF、TXT、MD、Word、Excel</p>
          <input
            :ref="(el) => { fileInput = el as HTMLInputElement }"
            type="file"
            style="display: none"
            accept=".pdf,.txt,.md,.docx,.xlsx,.xls"
            @change="handleUpload"
          />
        </div>
        <LoadingSpinner v-if="uploading" text="上传中..." />
        <div v-if="result && !processing" class="mt-4 p-3 rounded-lg" :class="result.startsWith('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' : 'bg-red-900/30 border border-red-500/30 text-red-400'" v-html="result"></div>
      </div>

      <div class="card p-6">
        <h3 class="font-semibold text-lg mb-4 text-white">目录批量处理</h3>
        <p class="text-sm text-slate-400 mb-4">输入文档目录路径，自动批量处理并向量化</p>
        <div class="space-y-4">
          <div>
            <label class="text-sm text-slate-400 mb-2 block">目录路径</label>
            <input
              v-model="directoryPath"
              type="text"
              class="input-glow"
              placeholder="例如: D:\projects\docs"
            />
          </div>
          <button class="btn-neon w-full" :disabled="processing" @click="processDir">
            <i :class="['fas', processing ? 'fa-spinner fa-spin' : 'fa-folder-open', 'mr-2']"></i>
            {{ processing ? '处理中...' : '开始处理' }}
          </button>
          <div v-if="result && !uploading" class="mt-4 p-3 rounded-lg" :class="result.startsWith('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' : 'bg-red-900/30 border border-red-500/30 text-red-400'" v-html="result"></div>
        </div>
      </div>
    </div>
  </div>
</template>
