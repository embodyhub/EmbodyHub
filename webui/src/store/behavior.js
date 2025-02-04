import axios from 'axios'

export default {
  namespaced: true,
  
  state: {
    patterns: {},
    performanceMetrics: {},
    predictions: {},
    recommendations: [],
    loading: false,
    error: null
  },

  mutations: {
    SET_PATTERNS(state, patterns) {
      state.patterns = patterns
    },
    SET_PERFORMANCE_METRICS(state, metrics) {
      state.performanceMetrics = metrics
    },
    SET_PREDICTIONS(state, predictions) {
      state.predictions = predictions
    },
    SET_RECOMMENDATIONS(state, recommendations) {
      state.recommendations = recommendations
    },
    SET_LOADING(state, loading) {
      state.loading = loading
    },
    SET_ERROR(state, error) {
      state.error = error
    }
  },

  actions: {
    async fetchAnalysisData({ commit }) {
      commit('SET_LOADING', true)
      commit('SET_ERROR', null)
      
      try {
        const response = await axios.get('/api/behavior/analysis')
        const data = response.data

        commit('SET_PATTERNS', data.patterns || {})
        commit('SET_PERFORMANCE_METRICS', data.performance_metrics || {})
        commit('SET_PREDICTIONS', data.predictions || {})
        commit('SET_RECOMMENDATIONS', data.recommendations || [])
      } catch (error) {
        commit('SET_ERROR', error.message)
        console.error('获取行为分析数据失败:', error)
      } finally {
        commit('SET_LOADING', false)
      }
    },

    async exportReport() {
      try {
        const response = await axios.get('/api/behavior/report', { responseType: 'blob' })
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `behavior_report_${new Date().toISOString()}.pdf`)
        document.body.appendChild(link)
        link.click()
        link.remove()
      } catch (error) {
        console.error('导出报告失败:', error)
        throw error
      }
    }
  },

  getters: {
    isLoading: state => state.loading,
    hasError: state => state.error !== null,
    errorMessage: state => state.error,
    patternsList: state => Object.entries(state.patterns).map(([key, value]) => ({
      name: key,
      ...value
    })),
    recentPerformance: state => state.performanceMetrics?.recent_performance || {},
    predictionAccuracy: state => state.predictions?.recent_performance?.accuracy_trend || {}
  }
}