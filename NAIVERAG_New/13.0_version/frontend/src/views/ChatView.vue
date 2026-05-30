<script setup lang="ts">
import { ref } from 'vue'
import { sendQuery } from '@/api/query'
import ChatMessage from '@/components/ChatMessage.vue'
import SourceCard from '@/components/SourceCard.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import type { ChatMessage as ChatMessageType, SourceItemData } from '@/types/chat'

const messages = ref<ChatMessageType[]>([])
const isProcessing = ref(false)
const question = ref('')
const k = ref(4)
const sources = ref<SourceItemData[]>([])

function addMessage(role: 'user' | 'assistant', content: string, meta?: Record<string, unknown>) {
  messages.value.push({
    id: Date.now().toString(),
    role,
    content,
    timestamp: new Date().toISOString(),
    ...meta,
  })
}

async function askQuestion() {
  const q = question.value.trim()
  if (!q || isProcessing.value) return

  addMessage('user', q)
  question.value = ''
  isProcessing.value = true
  sources.value = []

  try {
    const data = await sendQuery({
      question: q,
      k: k.value,
      score_threshold: 0.3,
      include_sources: true,
    })

    if (data.success) {
      addMessage('assistant', data.answer, {
        metadata: {
          processing_time: data.processing_time,
          model_used: data.model_used,
          retrieval_mode: data.retrieval_mode,
          source_count: data.source_count,
        },
      })
      if (data.sources?.length) {
        sources.value = data.sources.map((s) => ({
          content: s.content,
          score: s.score,
          rank: s.rank,
        }))
      }
    } else {
      addMessage('assistant', '处理失败，请稍后重试。')
    }
  } catch (e: unknown) {
    const errMsg = e instanceof Error ? e.message : '网络请求失败'
    addMessage('assistant', `错误: ${errMsg}`)
  } finally {
    isProcessing.value = false
  }
}

function onKeyPress(e: KeyboardEvent) {
  if (e.key === 'Enter') askQuestion()
}
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-semibold text-white flex items-center gap-2">
        <i class="fas fa-comments text-neon-cyan"></i>
        智能问答
      </h2>
      <div class="flex items-center gap-3">
        <label class="text-xs text-slate-400">返回数量:</label>
        <select v-model.number="k" class="input-glow !py-2 !w-20 text-sm">
          <option :value="2">2</option>
          <option :value="4">4</option>
          <option :value="6">6</option>
          <option :value="8">8</option>
          <option :value="10">10</option>
        </select>
      </div>
    </div>

    <div class="card flex-1 flex flex-col p-4">
      <div class="chat-container flex-1 mb-3">
        <div v-if="!messages.length" class="flex items-start gap-3 mb-4">
          <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-cyan to-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
            <i class="fas fa-robot text-xs text-white"></i>
          </div>
          <div class="chat-msg chat-assistant flex-1">
            <p class="text-sm">您好！我是您的智能文档助手，可以回答基于文档内容的问题。</p>
            <div class="text-xs text-slate-500 mt-2">System</div>
          </div>
        </div>
        <div v-for="msg in messages" :key="msg.id" class="flex items-start gap-3 mb-4">
          <div v-if="msg.role === 'assistant'" class="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-cyan to-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
            <i class="fas fa-robot text-xs text-white"></i>
          </div>
          <ChatMessage :message="msg" class="flex-1" />
          <div v-if="msg.role === 'user'" class="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 mt-1">
            <i class="fas fa-user text-xs text-white"></i>
          </div>
        </div>
        <LoadingSpinner v-if="isProcessing" text="正在思考中..." />
      </div>

      <div class="flex gap-2">
        <input
          v-model="question"
          type="text"
          class="input-glow flex-1"
          placeholder="输入您的问题..."
          :disabled="isProcessing"
          @keypress="onKeyPress"
        />
        <button class="btn-neon" :disabled="isProcessing" @click="askQuestion">
          <i class="fas fa-paper-plane mr-1"></i>发送
        </button>
      </div>
    </div>

    <SourceCard :sources="sources" />
  </div>
</template>
