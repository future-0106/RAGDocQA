import axios from 'axios'

const request = axios.create({
  baseURL: '/api',
  timeout: 180000,
  headers: {
    'Content-Type': 'application/json',
  },
})

request.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.error || err.message || '请求失败'
    console.error('[API Error]', msg)
    return Promise.reject(new Error(msg))
  }
)

export default request
