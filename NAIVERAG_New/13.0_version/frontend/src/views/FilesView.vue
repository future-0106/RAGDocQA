<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getFiles, deleteFile } from '@/api/files'

const files = ref<string[]>([])
const loading = ref(true)
const deleting = ref<string | null>(null)

async function loadFiles() {
  loading.value = true
  try {
    const data = await getFiles()
    if (data.success) {
      files.value = data.files || []
    }
  } catch {
    files.value = []
  } finally {
    loading.value = false
  }
}

async function removeFile(filename: string) {
  if (!confirm(`确定删除 ${filename} ?`)) return
  deleting.value = filename
  try {
    await deleteFile(filename)
    await loadFiles()
  } finally {
    deleting.value = null
  }
}

onMounted(loadFiles)
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-semibold text-white">
        <i class="fas fa-folder-open text-neon-cyan mr-3"></i>文件管理
      </h2>
      <button class="btn-glow" @click="loadFiles">
        <i class="fas fa-sync-alt mr-2"></i>刷新
      </button>
    </div>

    <div class="card p-6">
      <div v-if="loading" class="text-slate-500 text-center py-8">加载中...</div>
      <div v-else-if="!files.length" class="text-slate-500 text-center py-8">暂无文件</div>
      <div v-else class="space-y-3">
        <div
          v-for="file in files"
          :key="file"
          class="flex items-center justify-between p-4 bg-dark-800 rounded-lg"
          style="background: #0f0f18"
        >
          <div class="flex items-center gap-3">
            <i class="fas fa-file-alt text-2xl text-slate-500"></i>
            <span class="text-white">{{ file }}</span>
          </div>
          <button
            class="btn-danger"
            :disabled="deleting === file"
            @click="removeFile(file)"
          >
            <i v-if="deleting === file" class="fas fa-spinner fa-spin mr-1"></i>
            删除
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
