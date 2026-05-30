<script setup lang="ts">
import type { ChatMessage } from '@/types/chat'

const props = defineProps<{
  message: ChatMessage
}>()

function formatTime(ts: string) {
  try {
    return new Date(ts).toLocaleTimeString()
  } catch {
    return ''
  }
}
</script>

<template>
  <div :class="['chat-msg', message.role === 'user' ? 'chat-user' : 'chat-assistant']">
    <p class="text-sm whitespace-pre-wrap">{{ message.content }}</p>
    <div :class="['text-xs mt-2', message.role === 'user' ? 'opacity-70' : 'text-slate-500']">
      {{ formatTime(message.timestamp) }}
    </div>
  </div>
</template>
