import axios from 'axios'

const BASE_URL = '/api/behavior'

export const getBehaviorAnalysis = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/analysis`)
    return response.data
  } catch (error) {
    console.error('获取行为分析数据失败:', error)
    throw error
  }
}

export const getPerformanceMetrics = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/performance`)
    return response.data
  } catch (error) {
    console.error('获取性能指标数据失败:', error)
    throw error
  }
}

export const getPredictionData = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/predictions`)
    return response.data
  } catch (error) {
    console.error('获取预测数据失败:', error)
    throw error
  }
}

export const exportReport = async (format = 'json') => {
  try {
    const response = await axios.get(`${BASE_URL}/export`, {
      params: { format },
      responseType: 'blob'
    })
    
    const blob = new Blob([response.data], {
      type: response.headers['content-type']
    })
    
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `behavior_report.${format}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('导出报告失败:', error)
    throw error
  }
}