(function () {
  const statusEl = document.getElementById('status');
  let ws;
  let app;
  let currentModel;

  function initPixi() {
    app = new PIXI.Application({ resizeTo: window, backgroundAlpha: 0, antialias: true });
    window.PIXI = PIXI;
    document.body.appendChild(app.view);
    window.addEventListener('resize', fitModel);
  }

  function toFileUrl(p) {
    return 'file:///' + String(p).replace(/\\/g, '/');
  }

  function fitModel() {
    if (!currentModel) return;
    const w = app.renderer.width;
    const h = app.renderer.height;
    const sw = currentModel.width || 1;
    const sh = currentModel.height || 1;
    const s = Math.min(w / sw, h / sh);
    currentModel.scale.set(s);
    currentModel.x = w / 2;
    currentModel.y = h / 2;
    currentModel.anchor.set(0.5, 0.5);
  }

  async function loadModel(input) {
    try {
      if (statusEl) statusEl.textContent = `正在加载模型: ${input}`;
      let url = String(input);
      if (!/^https?:/i.test(url)) {
        url = toFileUrl(url);
      }
      const model = await PIXI.live2d.Live2DModel.from(url, {
        onError: (e) =\u003e {
          if (statusEl) statusEl.textContent = '模型资源加载失败，请检查控制台';
          console.error('Live2D Loader Error:', e);
        },
      });

      if (currentModel) {
        app.stage.removeChild(currentModel);
        currentModel.destroy({ children: true, texture: true, baseTexture: true });
      }

      currentModel = model;
      app.stage.addChild(model);
      fitModel();
      if (model.once) {
        model.once('ready', fitModel);
        model.once('load', fitModel);
      }
      try {
        const settings = model.internalModel && model.internalModel.settings;
        const motions = (settings && settings.motions) || {};
        const groups = Object.keys(motions);
        if (groups.length) {
          let group = groups[0];
          let idx = 0;
          const list = motions[group] || [];
          for (let i = 0; i < list.length; i++) {
            const f = list[i].File || list[i].file || '';
            if (String(f).toLowerCase().includes('idle')) { idx = i; break; }
          }
          model.internalModel.motionManager.startMotion(group, idx);
        }
      } catch {}
      if (statusEl) statusEl.textContent = '模型加载成功';
    } catch (e) {
      if (statusEl) statusEl.textContent = '加载模型时发生严重错误，请检查控制台';
      console.error('Failed to load model:', e);
    }
  }

  function playAction(name) {
    if (!currentModel) return;
    try {
      const groups = Object.keys(currentModel.internalModel.motionManager.definitions || {});
      if (name \u0026\u0026 currentModel.motion) {
        currentModel.motion(name);
        return;
      }
      if (groups.length) {
        currentModel.motion(groups[0]);
      }
    } catch (e) {
      console.error('Failed to play action:', name, e);
    }
  }

  function connect() {
    ws = new WebSocket('ws://localhost:3399/');
    ws.onopen = () =\u003e { if (statusEl) statusEl.textContent = '已连接 WebSocket'; };
    ws.onclose = () =\u003e { if (statusEl) statusEl.textContent = '连接断开，重试中...'; setTimeout(connect, 1500); };
    ws.onerror = (e) =\u003e { if (statusEl) statusEl.textContent = 'WebSocket 错误'; console.error(e); };
    ws.onmessage = async (ev) =\u003e {
      try {
        const msg = JSON.parse(ev.data);
        console.log('Received message:', msg);
        if (msg.type === 'MODEL_CHANGE' \u0026\u0026 (msg.url || msg.path)) {
          await loadModel(msg.url || msg.path);
        } else if (msg.type === 'ACTION') {
          playAction(msg.name);
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = '处理消息时出错';
        console.error('Failed to handle message:', ev.data, e);
      }
    };
  }

  function getDefaultModel() {
    const params = new URLSearchParams(location.search);
    const qp = params.get('model');
    if (qp) return qp;
    const hash = location.hash ? location.hash.slice(1) : '';
    if (hash) return decodeURIComponent(hash);
    return '../models/banrenma_2/banrenma_2.model3.json';
  }

  initPixi();
  connect();
  loadModel(getDefaultModel());
})();