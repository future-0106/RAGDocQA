<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getHistory, clearHistory } from '@/api/history'
import type { HistoryItem } from '@/types/api'

const history = ref<HistoryItem[]>([])
const loading = ref(true)

async function loadHistory() {
  loading.value = true
  try {
    const data = await getHistory(50)
    if (data.success) {
      history.value = data.history || []
    }
  } finally {
    loading.value = false
  }
}

async function handleClear() {
  if (!confirm('清空所有历史?')) return
  try {
    await clearHistory()
    await loadHistory()
  } catch {
    // ignore
  }
}

onMounted(loadHistory)
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-semibold text-white">
        <i class="fas fa-history text-neon-cyan mr-3"></i>查询历史
      </h2>
      <button class="btn-danger" @click="handleClear">
        <i class="fas fa-trash-alt mr-2"></i>清空历史
      </button>
    </div>

    <div class="card p-6">
      <div v-if="loading" class="text-slate-500 text-center py-8">加载中...</div>
      <div v-else-if="!history.length" class="text-slate-500 text-center py-8">暂无历史记录</div>
      <div v-else class="space-y-3">
        <div
          v-for="(item, idx) in history"
          :key="idx"
          class="p-4 rounded-lg"
          style="background: #0f0f18"
        >
          <div class="text-white mb-1">{{ item.question }}</div>
          <div class="text-xs text-slate-500">
            {{ item.timestamp }}
            <span v-if="item.model" class="ml-2 text-neon-cyan">[{{ item.model }}]</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
