/**
 * Panthera-HT Teach Pendant — RealMan-style browser teaching UI
 * Dual model: solid = live robot, red ghost = editable target
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import URDFLoader from 'urdf-loader'
import { io } from 'socket.io-client'

const DEG = 180 / Math.PI
const RAD = Math.PI / 180
const JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
const URDF_URL = '/Panthera-HT_description/urdf/Panthera-HT_description_follower.urdf'
const PKG_ROOT = '/Panthera-HT_description/'

class TeachPendantApp {
  constructor() {
    this.socket = null
    this.connected = false
    this.demoMode = false
    this.controlMode = 'position'
    this.speedPct = 40
    this.stepDeg = 1
    this.jointLimits = JOINT_NAMES.map(() => ({ min: -Math.PI, max: Math.PI }))
    this.realPos = new Array(6).fill(0)
    this.ghostPos = new Array(6).fill(0)
    this.ee = { position: [0, 0, 0], euler: [0, 0, 0] }

    this.realRobot = null
    this.ghostRobot = null
    this.world = null // URDF Z-up → Three.js Y-up
    this.scene = null
    this.camera = null
    this.renderer = null
    this.controls = null
    this.raycaster = new THREE.Raycaster()
    this.pointer = new THREE.Vector2()
    this.dragJoint = null
    this.dragPlane = new THREE.Plane()
    this.dragPivot = new THREE.Vector3()
    this._prevHit = new THREE.Vector3()
    this._tmp = new THREE.Vector3()
    this._tmp2 = new THREE.Vector3()
    this.jogTimer = null
    this.applyHolding = false
    this.ghostSynced = false

    this.els = {
      canvas: document.getElementById('canvas'),
      connDot: document.getElementById('connDot'),
      connText: document.getElementById('connText'),
      modeText: document.getElementById('modeText'),
      speedSlider: document.getElementById('speedSlider'),
      speedText: document.getElementById('speedText'),
      jogList: document.getElementById('jogList'),
      jointTable: document.getElementById('jointTable'),
      eeBlock: document.getElementById('eeBlock'),
      toast: document.getElementById('toast'),
      btnApply: document.getElementById('btnApply'),
      btnRestore: document.getElementById('btnRestore'),
      btnGravity: document.getElementById('btnGravity'),
      btnPosition: document.getElementById('btnPosition'),
      btnEstop: document.getElementById('btnEstop'),
      btnHome: document.getElementById('btnHome'),
      btnCopyJoints: document.getElementById('btnCopyJoints'),
      stepChips: document.getElementById('stepChips'),
    }
  }

  async init() {
    this.buildJogUI()
    this.bindUI()
    this.initScene()
    this.animate()
    try {
      await this.loadRobots()
      this.toast('Model loaded')
    } catch (e) {
      console.error(e)
      this.toast('Model load failed: ' + e.message)
    }
    this.connectSocket()
  }

  buildJogUI() {
    this.els.jogList.innerHTML = JOINT_NAMES.map((_, i) => `
      <div class="jog-row" data-joint="${i}">
        <div class="name">J${i + 1}</div>
        <div class="val" data-role="live">0.0°</div>
        <button class="jog-btn" data-dir="-1" data-joint="${i}">−</button>
        <div class="val" data-role="ghost" style="color:var(--ghost)">0.0°</div>
        <button class="jog-btn" data-dir="1" data-joint="${i}">+</button>
      </div>
    `).join('')

    this.els.jointTable.innerHTML = JOINT_NAMES.map((_, i) => `
      <tr>
        <td>J${i + 1}</td>
        <td class="num" data-live="${i}">0.00</td>
        <td class="num ghost-col" data-ghost="${i}">0.00</td>
      </tr>
    `).join('')
  }

  bindUI() {
    this.els.speedSlider.addEventListener('input', () => {
      this.speedPct = Number(this.els.speedSlider.value)
      this.els.speedText.textContent = `${this.speedPct}%`
    })

    this.els.stepChips.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-step]')
      if (!btn) return
      this.stepDeg = Number(btn.dataset.step)
      this.els.stepChips.querySelectorAll('.chip').forEach((c) => c.classList.toggle('active', c === btn))
    })

    this.els.jogList.addEventListener('pointerdown', (e) => {
      const btn = e.target.closest('.jog-btn')
      if (!btn) return
      e.preventDefault()
      const joint = Number(btn.dataset.joint)
      const dir = Number(btn.dataset.dir)
      this.startJog(joint, dir)
      const stop = () => this.stopJog()
      window.addEventListener('pointerup', stop, { once: true })
      window.addEventListener('pointercancel', stop, { once: true })
    })

    this.els.btnApply.addEventListener('click', () => this.applyOnce())
    // E-stop can still interrupt motion
    this.els.btnRestore.addEventListener('click', () => this.restoreGhost())
    this.els.btnGravity.addEventListener('click', () => this.setMode('gravity_comp'))
    this.els.btnPosition.addEventListener('click', () => this.setMode('position'))
    this.els.btnEstop.addEventListener('click', () => {
      this.applyHolding = false
      this.els.btnApply.classList.remove('running')
      this.emit('stop')
      this.toast('E-stop: target locked to current pose')
    })
    this.els.btnHome.addEventListener('click', () => {
      this.emit('reset_all')
      this.toast('Homing...')
    })
    this.els.btnCopyJoints.addEventListener('click', async () => {
      const deg = this.realPos.map((r) => +(r * DEG).toFixed(3))
      try {
        await navigator.clipboard.writeText(JSON.stringify(deg))
        this.toast('Copied live joint angles (deg)')
      } catch {
        this.toast(JSON.stringify(deg))
      }
    })
  }

  initScene() {
    const canvas = this.els.canvas
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0a0a0a)

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50)
    this.camera.position.set(0.9, 0.45, 0.9)

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.shadowMap.enabled = true

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.target.set(0, 0.2, 0)
    this.controls.enableDamping = true
    this.controls.update()

    // Match Host SceneManager: wrap URDF in world rotated -90° about X (Z-up → Y-up)
    this.world = new THREE.Object3D()
    this.world.rotation.set(-Math.PI / 2, 0, 0)
    this.scene.add(this.world)

    const hemi = new THREE.HemisphereLight(0xffffff, 0x222228, 0.7)
    this.scene.add(hemi)
    const dir = new THREE.DirectionalLight(0xffffff, 1.0)
    dir.position.set(2, 3, 1)
    dir.castShadow = true
    this.scene.add(dir)
    const fill = new THREE.DirectionalLight(0x4da3ff, 0.25)
    fill.position.set(-2, 1, -1)
    this.scene.add(fill)

    const grid = new THREE.GridHelper(2.0, 20, 0x3a3a3c, 0x2c2c2e)
    grid.position.y = 0
    this.scene.add(grid)

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(3, 3),
      new THREE.MeshStandardMaterial({ color: 0x111113, roughness: 0.95, metalness: 0 })
    )
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.001
    ground.receiveShadow = true
    this.scene.add(ground)

    canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e))
    window.addEventListener('pointermove', (e) => this.onPointerMove(e))
    window.addEventListener('pointerup', () => this.onPointerUp())
    window.addEventListener('resize', () => this.resize())
    this.resize()
  }

  resize() {
    const canvas = this.els.canvas
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    if (!w || !h) return
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
  }

  async loadRobots() {
    const loader = this.createUrdfLoader()
    this.realRobot = await this.loadUrdf(loader, URDF_URL)
    this.ghostRobot = await this.loadUrdf(loader, URDF_URL)

    this.styleRobot(this.realRobot, { opacity: 0.92, ghost: false })
    this.styleRobot(this.ghostRobot, { opacity: 0.45, ghost: true })

    this.world.add(this.realRobot)
    this.world.add(this.ghostRobot)
    this.ghostRobot.renderOrder = 2
    this.realRobot.renderOrder = 1
  }

  createUrdfLoader() {
    const loader = new URDFLoader()
    const stl = new STLLoader()
    loader.packages = { 'Panthera-HT_description': PKG_ROOT.replace(/\/$/, '') }
    loader.loadMeshCb = (path, manager, onComplete) => {
      let url = path
      if (url.startsWith('package://Panthera-HT_description/')) {
        url = PKG_ROOT + url.slice('package://Panthera-HT_description/'.length)
      } else if (url.startsWith('package://')) {
        url = '/' + url.slice('package://'.length)
      }
      stl.load(
        url,
        (geom) => {
          const mat = new THREE.MeshPhongMaterial({ color: 0x9aa3ad, flatShading: false })
          const mesh = new THREE.Mesh(geom, mat)
          mesh.castShadow = true
          onComplete(mesh)
        },
        undefined,
        (err) => onComplete(null, err)
      )
    }
    return loader
  }

  loadUrdf(loader, url) {
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (robot) => resolve(robot),
        undefined,
        (err) => reject(err || new Error('URDF load failed'))
      )
    })
  }

  styleRobot(robot, { opacity, ghost }) {
    robot.traverse((obj) => {
      if (!obj.isMesh) return
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
      mats.forEach((m, idx) => {
        if (!m) return
        const mat = m.clone()
        mat.transparent = opacity < 1
        mat.opacity = opacity
        mat.depthWrite = opacity >= 0.99
        if (ghost) {
          mat.color = new THREE.Color(0xe53935)
          mat.emissive = new THREE.Color(0x4a0000)
          mat.emissiveIntensity = 0.15
        }
        if (Array.isArray(obj.material)) obj.material[idx] = mat
        else obj.material = mat
      })
    })
  }

  connectSocket() {
    const url = `${window.location.protocol}//${window.location.hostname}:5000`
    this.socket = io(url, { transports: ['websocket', 'polling'] })

    this.socket.on('connect', () => {
      this.connected = true
      this.updateConnUI()
      this.toast('Backend connected')
    })
    this.socket.on('disconnect', () => {
      this.connected = false
      this.updateConnUI()
    })
    this.socket.on('connect_error', () => {
      this.connected = false
      this.updateConnUI('Connection failed; start backend.sh first')
    })
    this.socket.on('config', (cfg) => {
      this.demoMode = !!cfg.demo_mode
      this.controlMode = cfg.control_mode || 'position'
      if (Array.isArray(cfg.joints)) {
        const arms = cfg.joints.filter((j) => j.kind !== 'gripper')
        arms.slice(0, 6).forEach((j, i) => {
          this.jointLimits[i] = { min: j.min, max: j.max }
        })
      }
      this.updateConnUI()
      this.updateModeUI()
    })
    this.socket.on('robot_state', (state) => this.onState(state))
    this.socket.on('mode_changed', (data) => {
      this.controlMode = data.mode
      this.updateModeUI()
    })
  }

  emit(event, data) {
    if (this.socket && this.connected) this.socket.emit(event, data)
  }

  onState(state) {
    if (Array.isArray(state.positions)) {
      for (let i = 0; i < 6; i++) this.realPos[i] = state.positions[i] ?? this.realPos[i]
      this.applyJoints(this.realRobot, this.realPos)
      if (!this.ghostSynced) {
        for (let i = 0; i < 6; i++) this.ghostPos[i] = this.realPos[i]
        this.applyJoints(this.ghostRobot, this.ghostPos)
        this.ghostSynced = true
      }
    }
    if (state.ee_position) this.ee.position = state.ee_position
    if (state.ee_euler) this.ee.euler = state.ee_euler
    if (state.control_mode) {
      this.controlMode = state.control_mode
      this.updateModeUI()
    }
    // Clear Apply running style when close to ghost target
    let diff = 0
    for (let i = 0; i < 6; i++) diff = Math.max(diff, Math.abs(this.ghostPos[i] - this.realPos[i]))
    if (diff < 0.03) this.els.btnApply.classList.remove('running')
    this.refreshReadouts()
  }

  applyJoints(robot, positionsRad) {
    if (!robot) return
    JOINT_NAMES.forEach((name, i) => {
      if (typeof robot.setJointValue === 'function') {
        robot.setJointValue(name, positionsRad[i])
      } else if (robot.joints && robot.joints[name]) {
        robot.joints[name].setJointValue(positionsRad[i])
      }
    })
  }

  setGhostJoint(index, rad) {
    const lim = this.jointLimits[index]
    const clamped = Math.max(lim.min, Math.min(lim.max, rad))
    if (Math.abs(clamped - rad) > 1e-6) this.toast(`J${index + 1} limit hit, clamped`)
    this.ghostPos[index] = clamped
    this.applyJoints(this.ghostRobot, this.ghostPos)
    this.refreshReadouts()
  }

  restoreGhost() {
    for (let i = 0; i < 6; i++) this.ghostPos[i] = this.realPos[i]
    this.applyJoints(this.ghostRobot, this.ghostPos)
    this.refreshReadouts()
    this.toast('Ghost restored to live pose')
  }

  applyOnce() {
    if (this.controlMode !== 'position') {
      this.toast('Switch to Position mode before Apply')
      return
    }
    // Prompt to edit ghost first when almost identical to live pose
    let diff = 0
    for (let i = 0; i < 6; i++) diff = Math.max(diff, Math.abs(this.ghostPos[i] - this.realPos[i]))
    if (diff < 0.01) {
      this.toast('Drag the red ghost or use +/- first')
      return
    }
    const velocity = Math.max(0.15, 1.2 * (this.speedPct / 100))
    this.els.btnApply.classList.add('running')
    this.emit('move_all', { positions: [...this.ghostPos], velocity })
    this.toast(`Applying ghost target (speed ${this.speedPct}%)`)
    // Button style also clears on arrival; timeout is a fallback
    clearTimeout(this._applyTimer)
    this._applyTimer = setTimeout(() => {
      this.els.btnApply.classList.remove('running')
    }, 8000)
  }

  startApply() {
    this.applyOnce()
  }

  stopApply() {
    // Kept for E-stop path; single-click Apply no longer auto-stops
    this.applyHolding = false
    this.els.btnApply.classList.remove('running', 'holding')
  }

  startJog(joint, dir) {
    this.stopJog()
    const tick = () => {
      if (this.controlMode !== 'position') {
        this.toast('Jog requires Position mode')
        this.stopJog()
        return
      }
      const delta = dir * this.stepDeg * RAD
      const next = this.realPos[joint] + delta
      const lim = this.jointLimits[joint]
      const clamped = Math.max(lim.min, Math.min(lim.max, next))
      this.emit('move_joint', { joint, position: clamped })
      // Also preview on ghost so operator sees intent
      this.setGhostJoint(joint, clamped)
    }
    tick()
    this.jogTimer = setInterval(tick, 120)
  }

  stopJog() {
    if (this.jogTimer) {
      clearInterval(this.jogTimer)
      this.jogTimer = null
    }
  }

  setMode(mode) {
    this.emit('set_mode', { mode })
    this.controlMode = mode
    this.updateModeUI()
    if (mode === 'gravity_comp') this.toast('Gravity compensation: arm is free to drag')
    if (mode === 'position') this.toast('Back to Position mode')
  }

  updateConnUI(extra) {
    const { connDot, connText } = this.els
    connDot.classList.remove('on', 'demo')
    if (!this.connected) {
      connText.textContent = extra || 'Disconnected'
      return
    }
    if (this.demoMode) {
      connDot.classList.add('demo')
      connText.textContent = 'Connected · Demo'
    } else {
      connDot.classList.add('on')
      connText.textContent = 'Connected · Live'
    }
  }

  updateModeUI() {
    const map = {
      position: 'Position',
      gravity_comp: 'Gravity',
      gravity_friction: 'Gravity+Friction',
      impedance: 'Impedance',
    }
    this.els.modeText.textContent = map[this.controlMode] || this.controlMode
    this.els.btnGravity.classList.toggle('active', this.controlMode === 'gravity_comp')
    this.els.btnPosition.classList.toggle('active', this.controlMode === 'position')
  }

  refreshReadouts() {
    JOINT_NAMES.forEach((_, i) => {
      const live = (this.realPos[i] * DEG).toFixed(2)
      const ghost = (this.ghostPos[i] * DEG).toFixed(2)
      const liveCell = this.els.jointTable.querySelector(`[data-live="${i}"]`)
      const ghostCell = this.els.jointTable.querySelector(`[data-ghost="${i}"]`)
      if (liveCell) liveCell.textContent = live
      if (ghostCell) ghostCell.textContent = ghost
      const row = this.els.jogList.querySelector(`.jog-row[data-joint="${i}"]`)
      if (row) {
        row.querySelector('[data-role="live"]').textContent = `${live}°`
        row.querySelector('[data-role="ghost"]').textContent = `${ghost}°`
      }
    })
    const p = this.ee.position
    const e = this.ee.euler
    this.els.eeBlock.innerHTML =
      `x ${p[0]?.toFixed?.(3) ?? '—'}  y ${p[1]?.toFixed?.(3) ?? '—'}  z ${p[2]?.toFixed?.(3) ?? '—'}<br/>` +
      `rx ${(e[0] * DEG)?.toFixed?.(2) ?? '—'} ry ${(e[1] * DEG)?.toFixed?.(2) ?? '—'} rz ${(e[2] * DEG)?.toFixed?.(2) ?? '—'}`
  }

  /* ---- Ghost joint drag (link pick → parent revolute); works while connected ---- */
  onPointerDown(e) {
    if (!this.ghostRobot || e.button !== 0) return
    this.updatePointer(e)
    this.raycaster.setFromCamera(this.pointer, this.camera)
    // Prefer ghost hits so connected/live state never blocks teaching drag
    const hits = this.raycaster.intersectObject(this.ghostRobot, true)
    if (!hits.length) return

    const joint = this.findParentJoint(hits[0].object)
    if (!joint) return

    e.preventDefault()
    e.stopPropagation()
    this.controls.enabled = false
    this.dragJoint = joint
    joint.threeObject.getWorldPosition(this.dragPivot)
    this.camera.getWorldDirection(this._tmp)
    this.dragPlane.setFromNormalAndCoplanarPoint(this._tmp, hits[0].point)
    this._prevHit.copy(hits[0].point)
    this.toast(`Dragging ghost J${joint.index + 1}`)
  }

  onPointerMove(e) {
    if (!this.dragJoint) return
    this.updatePointer(e)
    this.raycaster.setFromCamera(this.pointer, this.camera)
    if (!this.raycaster.ray.intersectPlane(this.dragPlane, this._tmp)) return

    const jointObj = this.dragJoint.threeObject
    const axis = this.getJointAxisWorld(jointObj)
    // Project motion onto plane perpendicular to joint axis
    this._tmp2.copy(this._tmp).sub(this.dragPivot)
    const prev = this._prevHit.clone().sub(this.dragPivot)
    const a = prev.clone().normalize()
    const b = this._tmp2.clone().normalize()
    if (a.lengthSq() < 1e-8 || b.lengthSq() < 1e-8) return

    let delta = Math.atan2(a.clone().cross(b).dot(axis), a.dot(b))
    if (!Number.isFinite(delta)) return

    const idx = this.dragJoint.index
    this.setGhostJoint(idx, this.ghostPos[idx] + delta)
    this._prevHit.copy(this._tmp)
  }

  onPointerUp() {
    if (!this.dragJoint) return
    this.dragJoint = null
    this.controls.enabled = true
  }

  updatePointer(e) {
    const rect = this.els.canvas.getBoundingClientRect()
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
  }

  findParentJoint(obj) {
    let cur = obj
    while (cur && cur !== this.ghostRobot) {
      if (cur.isURDFJoint && cur.jointType !== 'fixed') {
        const name = cur.name
        const index = JOINT_NAMES.indexOf(name)
        if (index >= 0) return { name, index, threeObject: cur }
      }
      cur = cur.parent
    }
    return null
  }

  getJointAxisWorld(jointObj) {
    const axisLocal = jointObj.axis
      ? new THREE.Vector3(jointObj.axis.x, jointObj.axis.y, jointObj.axis.z)
      : new THREE.Vector3(0, 0, 1)
    const q = new THREE.Quaternion()
    jointObj.getWorldQuaternion(q)
    return axisLocal.applyQuaternion(q).normalize()
  }

  animate() {
    requestAnimationFrame(() => this.animate())
    this.controls?.update()
    this.renderer?.render(this.scene, this.camera)
  }

  toast(msg) {
    const el = this.els.toast
    el.textContent = msg
    el.classList.add('show')
    clearTimeout(this._toastTimer)
    this._toastTimer = setTimeout(() => el.classList.remove('show'), 1800)
  }
}

const app = new TeachPendantApp()
app.init()
