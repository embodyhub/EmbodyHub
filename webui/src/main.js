import { createApp } from 'vue'
import { createStore } from 'vuex'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import './assets/main.css'
import App from './App.vue'

// Router configuration
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Dashboard',
      component: () => import('./views/Dashboard.vue')
    },
    {
      path: '/models',
      name: 'Models',
      component: () => import('./views/Models.vue')
    },
    {
      path: '/monitor',
      name: 'Monitor',
      component: () => import('./views/Monitor.vue')
    },
    {
      path: '/behavior',
      name: 'BehaviorAnalysis',
      component: () => import('./views/BehaviorAnalysis.vue')
    }
  ]
})

// Vuex state management
const store = createStore({
  state() {
    return {
      models: [],
      monitorData: {},
      systemStatus: 'idle'
    }
  },
  mutations: {
    setModels(state, models) {
      state.models = models
    },
    updateMonitorData(state, data) {
      state.monitorData = data
    },
    setSystemStatus(state, status) {
      state.systemStatus = status
    }
  }
})

const app = createApp(App)

// Use plugins and router
app.use(ElementPlus)
app.use(store)
app.use(router)

// Mount application
app.mount('#app')