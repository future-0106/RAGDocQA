<script setup lang="ts">
import type { SourceItemData } from '@/types/chat'

defineProps<{
  sources: SourceItemData[]
}>()
</script>

<template>
  <div v-if="sources.length" class="card p-6 mt-6">
    <h3 class="font-semibold text-lg mb-4 text-white">
      <i class="fas fa-file-alt text-neon-cyan mr-2"></i>参考来源
    </h3>
    <div style="max-height: 300px; overflow-y: auto">
      <div
        v-for="(source, idx) in sources"
        :key="idx"
        class="source-item"
      >
        <div class="flex justify-between mb-2">
          <strong class="text-white">来源 {{ source.rank || idx + 1 }}</strong>
          <span class="badge badge-cyan">Score: {{ source.score?.toFixed(3) }}</span>
        </div>
        <div class="text-sm text-slate-400">
          {{ source.content?.substring(0, 200) }}...
        </div>
      </div>
    </div>
  </div>
</template>
