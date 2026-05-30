<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getModels, switchModel, switchEmbedding } from '@/api/models'
import ModelCard from '@/components/ModelCard.vue'
import type { ModelItem } from '@/types/api'

const llmModels = ref<ModelItem[]>([])
const embeddingModels = ref<ModelItem[]>([])
const loading = ref(true)
const switching = ref(false)

async function loadModels() {
  loading.value = true
  try {
    const data = await getModels()
    if (data.success) {
      llmModels.value = data.llm_models || []
      embeddingModels.value = data.embedding_models || []
    }
  } finally {
    loading.value = false
  }
}

async function handleSwitch(key: string, type: 'llm' | 'embedding') {
  if (!confirm(`切换到 ${key} ?`)) return
  switching.value = true
  try {
    if (type === 'llm') {
      await switchModel(key)
    } else {
      await switchEmbedding(key)
    }
    await loadModels()
  } finally {
    switching.value = false
  }
}

onMounted(loadModels)
</script>

<template>
  <div>
    <h2 class="text-2xl font-semibold text-white mb-6">
      <i class="fas fa-microchip text-neon-cyan mr-3"></i>模型管理
    </h2>

    <LoadingSpinner v-if="loading || switching" :text="switching ? '切换中...' : '加载中...'" />

    <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="card p-6">
        <h3 class="font-semibold text-lg mb-4 text-white">
          <i class="fas fa-brain" style="color: #a78bfa; margin-right: 8px"></i>LLM 模型
        </h3>
        <div v-if="llmModels.length" class="space-y-3">
          <ModelCard
            v-for="model in llmModels"
            :key="model.key"
            :model="model"
            @switch="(key) => handleSwitch(key, 'llm')"
          />
        </div>
        <p v-else class="text-slate-500">无可用模型</p>
      </div>

      <div class="card p-6">
        <h3 class="font-semibold text-lg mb-4 text-white">
          <i class="fas fa-layer-group" style="color: #a78bfa; margin-right: 8px"></i>嵌入模型
        </h3>
        <div v-if="embeddingModels.length" class="space-y-3">
          <ModelCard
            v-for="model in embeddingModels"
            :key="model.key"
            :model="model"
            @switch="(key) => handleSwitch(key, 'embedding')"
          />
        </div>
        <p v-else class="text-slate-500">无可用模型</p>
      </div>
    </div>
  </div>
</template>
