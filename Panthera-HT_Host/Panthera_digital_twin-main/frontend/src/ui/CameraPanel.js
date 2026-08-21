/**
 * CameraPanel - D435i color preview in the Host digital twin.
 * Stream starts when the panel is visible; closing it releases the camera.
 */
export class CameraPanel {
    constructor(panelManager = null) {
        this.panelManager = panelManager
        this.panel = null
        this.toggleBtn = null
        this.image = null
        this.overlay = null
        this.statusDot = null
        this.statusText = null
        this.retryTimer = null
        this.streaming = false
    }

    init() {
        this.panel = document.getElementById('floating-camera-panel')
        this.toggleBtn = document.getElementById('toggle-camera-panel')
        this.image = document.getElementById('camera-stream')
        this.overlay = document.getElementById('camera-overlay')
        this.statusDot = document.getElementById('camera-status-dot')
        this.statusText = document.getElementById('camera-status-text')
        if (!this.panel || !this.toggleBtn || !this.image) return

        this.panelManager?.registerPanel('floating-camera-panel')
        this.toggleBtn.classList.toggle('active', this.isVisible())

        this.toggleBtn.addEventListener('click', () => {
            const visible = this.panelManager
                ? this.panelManager.togglePanel('floating-camera-panel')
                : this.panel.style.display === 'none'
            if (!this.panelManager) {
                this.panel.style.display = visible ? 'flex' : 'none'
            }
            this.toggleBtn.classList.toggle('active', visible)
            if (visible) this.startStream()
            else this.stopStream()
        })

        const closeBtn = this.panel.querySelector('.panel-close-btn')
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (this.panelManager) {
                    this.panelManager.hidePanel('floating-camera-panel')
                } else {
                    this.panel.style.display = 'none'
                }
                this.toggleBtn.classList.remove('active')
                this.stopStream()
            })
        }

        this.image.addEventListener('load', () => {
            this.setStatus(true, 'D435i live')
            this.hideOverlay()
        })
        this.image.addEventListener('error', () => {
            this.setStatus(false, 'Camera unavailable')
            this.showOverlay('Waiting for D435i…')
            this.scheduleRetry()
        })

        if (this.isVisible()) {
            this.startStream()
        }
    }

    isVisible() {
        if (this.panelManager) {
            return this.panelManager.isPanelVisible('floating-camera-panel')
        }
        return this.panel && this.panel.style.display !== 'none'
    }

    serverUrl() {
        const input = document.getElementById('server-url')
        const value = input?.value?.trim()
        return value || 'http://localhost:5000'
    }

    startStream() {
        this.clearRetry()
        this.streaming = true
        this.showOverlay('Connecting D435i…')
        this.setStatus(false, 'Connecting…')
        this.image.src = `${this.serverUrl()}/api/camera/stream?t=${Date.now()}`
    }

    stopStream() {
        this.streaming = false
        this.clearRetry()
        this.image.removeAttribute('src')
        this.showOverlay('Camera stopped')
        this.setStatus(false, 'Stopped')
    }

    scheduleRetry() {
        this.clearRetry()
        if (!this.streaming || !this.isVisible()) return
        this.retryTimer = window.setTimeout(() => {
            if (this.streaming && this.isVisible()) this.startStream()
        }, 2500)
    }

    clearRetry() {
        if (this.retryTimer) {
            window.clearTimeout(this.retryTimer)
            this.retryTimer = null
        }
    }

    setStatus(connected, text) {
        if (this.statusDot) {
            this.statusDot.classList.toggle('connected', connected)
            this.statusDot.classList.toggle('disconnected', !connected)
        }
        if (this.statusText) this.statusText.textContent = text
    }

    showOverlay(text) {
        if (!this.overlay) return
        this.overlay.textContent = text
        this.overlay.style.display = 'flex'
    }

    hideOverlay() {
        if (this.overlay) this.overlay.style.display = 'none'
    }
}
