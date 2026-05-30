<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getStatus } from '@/api/status'
import StatCard from '@/components/StatCard.vue'
import type { SystemStatus } from '@/types/api'

const status = ref<SystemStatus | null>(null)
const loading = ref(true)

async function loadStatus() {
  loading.value = true
  try {
    const data = await getStatus()
    if (data.success) {
      status.value = data.data
    }
  } finally {
    loading.value = false
  }
}

onMounted(loadStatus)
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-semibold text-white">
        <i class="fas fa-chart-line text-neon-cyan mr-3"></i>系统状态
      </h2>
      <button class="btn-glow" @click="loadStatus">
        <i class="fas fa-sync-alt mr-2"></i>刷新
      </button>
    </div>

    <div v-if="loading && !status" class="text-slate-500 text-center py-8">加载中...</div>

    <template v-if="status">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard icon="fa-file-alt" :value="status.files?.count || 0" label="文档数量" color="cyan" />
        <StatCard icon="fa-database" :value="status.vector_store?.document_count ?? 0" label="向量数量" color="purple" icon-color="#a78bfa" />
        <StatCard icon="fa-microchip" :value="status.models?.llm ? 1 : 0" label="可用模型" color="green" icon-color="#10b981" />
        <StatCard icon="fa-history" :value="status.history?.count || 0" label="查询历史" color="orange" icon-color="#fbbf24" />
      </div>

      <div class="card p-6">
        <div class="grid grid-cols-2 gap-4">
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">LLM 模型</p>
            <p class="text-neon-cyan font-medium">{{ status.models?.llm }}</p>
          </div>
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">嵌入模型</p>
            <p class="text-neon-cyan font-medium" style="color: #a78bfa">{{ status.models?.embedding }}</p>
          </div>
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">设备</p>
            <p class="text-white font-medium">{{ status.device }}</p>
          </div>
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">改写模型</p>
            <p class="text-white font-medium">{{ status.models?.rewrite_llm }}</p>
          </div>
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">分块大小</p>
            <p class="text-white font-medium">{{ status.config?.chunk_size }}</p>
          </div>
          <div class="p-4 rounded-lg" style="background: #0f0f18">
            <p class="text-sm text-slate-500">相似度阈值</p>
            <p class="text-white font-medium">{{ status.config?.score_threshold }}</p>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
