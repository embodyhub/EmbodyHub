import { createStore } from 'vuex'
import models from './modules/models'

export default createStore({
  modules: {
    models
  },
  state() {
    return {
      monitorData: {},
      systemStatus: 'idle'
    }
  },
  mutations: {
    updateMonitorData(state, data) {
      state.monitorData = data
    },
    setSystemStatus(state, status) {
      state.systemStatus = status
    }
  },
  actions: {
    async updateSystemStatus({ commit }, status) {
      commit('setSystemStatus', status)
    }
  }
})