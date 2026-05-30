<script setup lang="ts">
import type { ModelItem } from '@/types/api'

const props = defineProps<{
  model: ModelItem
}>()

const emit = defineEmits<{
  switch: [key: string]
}>()

const isLocal = props.model.type === 'local'
const isApi = props.model.type === 'api'
</script>

<template>
  <div
    :class="['model-card', { active: model.is_current }]"
    @click="emit('switch', model.key)"
  >
    <div class="flex justify-between items-start">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <strong class="text-white truncate">{{ model.key }}</strong>
          <span v-if="model.is_current" class="badge badge-cyan">当前</span>
        </div>
        <p class="text-xs text-slate-400 mt-1 truncate">{{ model.description }}</p>
        <div class="flex gap-2 mt-2">
          <span v-if="isLocal" class="badge badge-purple">本地</span>
          <span v-else-if="isApi" class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981">云端</span>
        </div>
      </div>
    </div>
  </div>
</template>
