import axios from 'axios'

const state = {
  models: [],
  selectedModel: null,
  loading: false,
  error: null
}

const mutations = {
  SET_MODELS(state, models) {
    state.models = models
  },
  SET_SELECTED_MODEL(state, model) {
    state.selectedModel = model
  },
  SET_LOADING(state, loading) {
    state.loading = loading
  },
  SET_ERROR(state, error) {
    state.error = error
  },
  ADD_MODEL(state, model) {
    state.models.push(model)
  },
  UPDATE_MODEL(state, updatedModel) {
    const index = state.models.findIndex(m => m.id === updatedModel.id)
    if (index !== -1) {
      state.models.splice(index, 1, updatedModel)
    }
  },
  REMOVE_MODEL(state, modelId) {
    state.models = state.models.filter(m => m.id !== modelId)
  }
}

const actions = {
  async fetchModels({ commit }) {
    commit('SET_LOADING', true)
    try {
      const response = await axios.get('/api/models')
      commit('SET_MODELS', response.data)
    } catch (error) {
      commit('SET_ERROR', error.message)
    } finally {
      commit('SET_LOADING', false)
    }
  },

  async addModel({ commit }, modelData) {
    commit('SET_LOADING', true)
    try {
      const response = await axios.post('/api/models', modelData)
      commit('ADD_MODEL', response.data)
      return response.data
    } catch (error) {
      commit('SET_ERROR', error.message)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  async updateModel({ commit }, { id, data }) {
    commit('SET_LOADING', true)
    try {
      const response = await axios.put(`/api/models/${id}`, data)
      commit('UPDATE_MODEL', response.data)
      return response.data
    } catch (error) {
      commit('SET_ERROR', error.message)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  async removeModel({ commit }, id) {
    commit('SET_LOADING', true)
    try {
      await axios.delete(`/api/models/${id}`)
      commit('REMOVE_MODEL', id)
    } catch (error) {
      commit('SET_ERROR', error.message)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  async toggleModelStatus({ commit }, { id, active }) {
    commit('SET_LOADING', true)
    try {
      const response = await axios.post(`/api/models/${id}/toggle`, { active })
      commit('UPDATE_MODEL', response.data)
      return response.data
    } catch (error) {
      commit('SET_ERROR', error.message)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}