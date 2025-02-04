import { getBehaviorAnalysis, getPerformanceMetrics, getPredictionData } from '@/api/behavior'

const state = {
  overviewStats: {
    totalBehaviors: 0,
    avgResponseTime: 0,
    avgReward: 0,
    predictionAccuracy: 0
  },
  patternDistribution: [],
  performanceTrend: {
    timestamps: [],
    values: []
  },
  predictionData: []
}

const mutations = {
  SET_OVERVIEW_STATS(state, stats) {
    state.overviewStats = stats
  },
  SET_PATTERN_DISTRIBUTION(state, distribution) {
    state.patternDistribution = distribution
  },
  SET_PERFORMANCE_TREND(state, trend) {
    state.performanceTrend = trend
  },
  SET_PREDICTION_DATA(state, data) {
    state.predictionData = data
  }
}

const actions = {
  async fetchBehaviorAnalysis({ commit }) {
    try {
      const data = await getBehaviorAnalysis()
      commit('SET_OVERVIEW_STATS', {
        totalBehaviors: data.total_behaviors || 0,
        avgResponseTime: data.avg_response_time || 0,
        avgReward: data.avg_reward || 0,
        predictionAccuracy: data.prediction_accuracy || 0
      })
      commit('SET_PATTERN_DISTRIBUTION', data.patterns || [])
    } catch (error) {
      console.error('获取行为分析数据失败:', error)
      throw error
    }
  },

  async fetchPerformanceMetrics({ commit }) {
    try {
      const data = await getPerformanceMetrics()
      commit('SET_PERFORMANCE_TREND', {
        timestamps: data.timestamps || [],
        values: data.values || []
      })
    } catch (error) {
      console.error('获取性能指标数据失败:', error)
      throw error
    }
  },

  async fetchPredictionData({ commit }) {
    try {
      const data = await getPredictionData()
      commit('SET_PREDICTION_DATA', data || [])
    } catch (error) {
      console.error('获取预测数据失败:', error)
      throw error
    }
  }
}

const getters = {
  formattedOverviewStats: state => ({
    totalBehaviors: state.overviewStats.totalBehaviors,
    avgResponseTime: `${state.overviewStats.avgResponseTime.toFixed(2)}ms`,
    avgReward: state.overviewStats.avgReward.toFixed(3),
    predictionAccuracy: `${(state.overviewStats.predictionAccuracy * 100).toFixed(1)}%`
  }),

  patternChartData: state => state.patternDistribution.map(pattern => ({
    name: pattern.name,
    value: pattern.percentage
  })),

  performanceChartData: state => ({
    timestamps: state.performanceTrend.timestamps,
    values: state.performanceTrend.values
  })
}

export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
}