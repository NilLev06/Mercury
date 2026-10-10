'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const app = { maps: [], selected: null, arena: { half: 3, obstacle_radius: .28, wheelbase: .18, track: .36 }, state: null, online: false, busy: false, cards: new Map(), error: '', loaded: false };
  const active = () => ['starting', 'running', 'stopping'].includes(app.state?.status);
  const groupLabel = value => ({ old: 'Базовые карты', complex32: 'Сложные · 32', independent: 'Независимые · 192' })[value] || value;
  const tierLabel = value => ({ simple: 'Базовая', complex: 'Сложная' })[value] || value;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 2, suffix = '') => finite(value) ? value.toFixed(digits) + suffix : '—';
  function outcome(report) {
    if (!report) return { kind: 'unknown', label: 'Нет истории проверки' };
    if (report.cancelled) return { kind: 'stopped', label: 'Остановлено' };
    if (report.success === true) {
      if (!finite(report.contacts)) return { kind: 'arrived', label: 'Прибытие · контакты неизвестны' };
      return report.contacts === 0 ? { kind: 'clean', label: 'Чистое прибытие' } : { kind: 'contact', label: 'Прибытие с контактами' };
    }
    if (report.success === false) return { kind: 'timeout', label: 'Лимит времени · без прибытия' };
    return { kind: 'unknown', label: 'Результат без оценки прибытия' };
  }
  function resultDetail(report) {
    if (!report) return 'Нет подтверждённого результата для этой карты.';
    const details = [];
    if (finite(report.contacts)) details.push(`${report.contacts} контактов`);
    if (finite(report.final_goal_distance)) details.push(`${fmt(report.final_goal_distance)} м до цели`);
    if (finite(report.steps)) details.push(`${report.steps} шагов`);
    return details.join(' · ') || 'Подробная телеметрия отсутствует.';
  }
  function setConnection(online) {
    app.online = online;
    $('connection').textContent = online ? 'Локальная сессия' : 'Связь потеряна · повторяем';
    $('connection').className = `connection ${online ? 'online' : 'offline'}`;
  }
  function showError(message) { app.error = message || ''; $('error').textContent = app.error; $('error').hidden = !app.error; }
  async function request(path, options = {}) {
    const response = await fetch(`/api/gallery/${path}`, { cache: 'no-store', signal: AbortSignal.timeout(10000), ...options });
    let data;
    try { data = await response.json(); } catch { throw new Error(`Сервер вернул ответ без JSON (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Ошибка сервера: ${response.status}`);
    return data;
  }
  function fit(canvas, thumbnail = false) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth || (thumbnail ? 130 : 600);
    const height = canvas.clientHeight || (thumbnail ? 91 : 450);
    const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }
  function drawMap(canvas, map, frame = null, trail = [], thumbnail = false) {
    const { ctx, width, height } = fit(canvas, thumbnail);
    ctx.fillStyle = '#f0f3e9'; ctx.fillRect(0, 0, width, height);
    if (!map) return;
    const half = app.arena.half || 3, margin = thumbnail ? 8 : 34;
    const size = Math.min(width - 2 * margin, height - 2 * margin), scale = size / (half * 2);
    const left = (width - size) / 2, top = (height - size) / 2;
    const px = x => left + (x + half) * scale, py = y => top + (half - y) * scale;
    ctx.fillStyle = '#f8faf1'; ctx.fillRect(left, top, size, size);
    ctx.lineWidth = 1; ctx.strokeStyle = '#e0e6d5';
    for (let n = -Math.floor(half); n <= half; n++) { ctx.beginPath(); ctx.moveTo(px(n), top); ctx.lineTo(px(n), top + size); ctx.moveTo(left, py(n)); ctx.lineTo(left + size, py(n)); ctx.stroke(); }
    ctx.strokeStyle = '#bccab0'; ctx.strokeRect(left, top, size, size);
    if (!thumbnail) {
      ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = '#819073'; ctx.textAlign = 'center';
      for (let n = -Math.floor(half); n <= half; n++) { ctx.fillText(n, px(n), top + size + 18); }
      ctx.textAlign = 'right'; for (let n = -Math.floor(half); n <= half; n++) ctx.fillText(n, left - 9, py(n) + 3);
      ctx.fillText('y', left - 9, top - 12); ctx.textAlign = 'left'; ctx.fillText('x', left + size + 13, top + size + 3);
    }
    for (const obstacle of map.obstacles || []) {
      const [x, y] = obstacle.position;
      if (!finite(x) || !finite(y)) continue;
      const radius = app.arena.obstacle_radius * scale;
      const rgb = (obstacle.colour || [.55, .65, .45]).map(v => Math.round(Math.max(0, Math.min(1, v)) * 255));
      ctx.beginPath(); ctx.arc(px(x), py(y), radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${rgb.join(',')})`; ctx.fill(); ctx.strokeStyle = '#58684c'; ctx.lineWidth = thumbnail ? .6 : 1; ctx.stroke();
      if (!thumbnail) { ctx.beginPath(); ctx.arc(px(x) - radius * .2, py(y) - radius * .25, radius * .55, Math.PI, Math.PI * 1.6); ctx.strokeStyle = '#ffffff55'; ctx.stroke(); }
    }
    if (Array.isArray(map.goal)) {
      const [gx, gy] = map.goal;
      if (!thumbnail) { ctx.beginPath(); ctx.arc(px(gx), py(gy), .5 * scale, 0, Math.PI * 2); ctx.fillStyle = '#d19b6416'; ctx.fill(); ctx.strokeStyle = '#ba895c66'; ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]); }
      ctx.beginPath(); ctx.arc(px(gx), py(gy), Math.max(3, .16 * scale), 0, Math.PI * 2); ctx.fillStyle = '#bc734b'; ctx.fill();
      if (!thumbnail) { ctx.fillStyle = '#895d40'; ctx.font = '10px -apple-system, sans-serif'; ctx.textAlign = 'center'; ctx.fillText('ЦЕЛЬ', px(gx), py(gy) - 15); }
    }
    if (!thumbnail && trail.length) {
      ctx.save(); ctx.beginPath(); ctx.rect(left, top, size, size); ctx.clip(); ctx.beginPath();
      let begun = false;
      for (const point of trail) { if (!Array.isArray(point) || !finite(point[0]) || !finite(point[1])) continue; begun ? ctx.lineTo(px(point[0]), py(point[1])) : ctx.moveTo(px(point[0]), py(point[1])); begun = true; }
      ctx.lineWidth = 2; ctx.strokeStyle = '#6a8d61'; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke(); ctx.restore();
    }
    const pose = frame?.pose || map.pose;
    if (Array.isArray(pose) && pose.every(finite)) {
      const [x, y, yaw] = pose;
      const centreX = x + app.arena.wheelbase / 2 * Math.cos(yaw), centreY = y + app.arena.wheelbase / 2 * Math.sin(yaw);
      ctx.save(); ctx.translate(px(centreX), py(centreY)); ctx.rotate(-yaw);
      const bodyLength = Math.max(thumbnail ? 7 : 13, (app.arena.wheelbase + .17) * scale), bodyWidth = Math.max(thumbnail ? 5 : 10, app.arena.track * scale);
      ctx.fillStyle = '#234a36'; ctx.fillRect(-bodyLength / 2, -bodyWidth / 2, bodyLength, bodyWidth);
      ctx.fillStyle = '#b6d098'; ctx.fillRect(bodyLength * .05, -bodyWidth * .32, bodyLength * .25, bodyWidth * .64);
      ctx.beginPath(); ctx.moveTo(bodyLength / 2 + 7, 0); ctx.lineTo(bodyLength / 2 + 1, -3); ctx.lineTo(bodyLength / 2 + 1, 3); ctx.closePath(); ctx.fillStyle = '#234a36'; ctx.fill(); ctx.restore();
    }
  }
  function filteredMaps() {
    const query = $('search').value.trim().toLocaleLowerCase('ru');
    return app.maps.filter(map => {
      const text = `${map.id} ${map.name} ${map.title} ${map.group} ${map.tier}`.toLocaleLowerCase('ru');
      const kind = outcome(map.known_result).kind;
      return (!query || text.includes(query)) && ($('group').value === 'all' || map.group === $('group').value) && ($('outcome').value === 'all' || ($('outcome').value === 'session' ? !!app.state?.history?.[map.id] : $('outcome').value === kind));
    });
  }
  function renderCards() {
    const focusedCard = $('cards').contains(document.activeElement) ? document.activeElement.closest('.map-card') : null;
    const maps = filteredMaps(); const fragment = document.createDocumentFragment();
    for (const map of maps) {
      let card = app.cards.get(map.id);
      if (!card) {
        card = document.createElement('button'); card.type = 'button'; card.className = 'map-card';
        const canvas = document.createElement('canvas'); canvas.setAttribute('aria-hidden', 'true');
        const body = document.createElement('div'); body.className = 'card-text';
        const title = document.createElement('strong'); title.textContent = map.title || map.name || map.id;
        const detail = document.createElement('small'); detail.textContent = `${groupLabel(map.group)} · ${tierLabel(map.tier)}`;
        const badge = document.createElement('span'); badge.className = 'badge';
        const session = document.createElement('small'); session.className = 'session-badge';
        body.append(title, detail, badge, session); card.append(canvas, body);
        card.addEventListener('click', () => { app.selected = map; renderCards(); renderView(); });
        app.cards.set(map.id, card);
      }
      const known = outcome(map.known_result);
      card.querySelector('.badge').textContent = known.label;
      card.querySelector('.badge').className = `badge ${known.kind}`;
      const session = app.state?.history?.[map.id];
      card.querySelector('.session-badge').textContent = session ? `Сеанс: ${outcome(session).label.toLocaleLowerCase('ru')}` : '';
      card.setAttribute('aria-pressed', String(app.selected?.id === map.id));
      card.classList.toggle('active', active() && app.state?.map_id === map.id);
      fragment.append(card);
    }
    $('cards').replaceChildren(fragment);
    if (focusedCard?.isConnected) focusedCard.focus({ preventScroll: true });
    for (const map of maps) drawMap(app.cards.get(map.id).querySelector('canvas'), map, null, [], true);
    $('matches').textContent = `${maps.length} из ${app.maps.length}`;
    $('empty-catalogue').hidden = maps.length !== 0;
  }
  function viewData() {
    const state = app.state;
    const showRun = !!state?.map_id && (active() || state.map_id === app.selected?.id);
    return { map: showRun ? app.maps.find(m => m.id === state.map_id) || app.selected : app.selected, frame: showRun ? state.frame : null, trail: showRun ? state.trail || [] : [], showRun };
  }
  function drawActivity(frame) {
    const { ctx, width, height } = fit($('activity'));
    const values = frame?.activity;
    if (!Array.isArray(values) || values.length !== 255 || !values.every(finite)) { $('activity-note').textContent = 'Нет данных активности в этом кадре.'; return; }
    const max = Math.max(...values.map(Math.abs), .000001);
    const columns = 17, rows = 15;
    values.forEach((value, i) => {
      const intensity = Math.min(1, Math.abs(value) / max), base = [238, 240, 221], target = value < 0 ? [72, 93, 154] : [162, 91, 56];
      ctx.fillStyle = `rgb(${base.map((v, j) => Math.round(v + (target[j] - v) * intensity)).join(',')})`;
      ctx.fillRect(i % columns * width / columns + 1, Math.floor(i / columns) * height / rows + 1, width / columns - 2, height / rows - 2);
    });
    $('activity-note').textContent = `255 состояний · диапазон ±${max.toFixed(3)}`;
  }
  let lastPreview = null;
  function renderView() {
    const state = app.state, selected = app.selected, { map, frame, trail, showRun } = viewData();
    if (!selected || !map) return;
    const isActive = active();
    $('map-title').textContent = map.title || map.name || map.id;
    $('map-meta').textContent = `${groupLabel(map.group)} · ${tierLabel(map.tier)} · ${map.id}`;
    $('view-mode').textContent = showRun && state?.frame ? 'НАБЛЮДЕНИЕ ЗА ПРОЕЗДОМ' : 'ПРЕДПРОСМОТР КАРТЫ';
    const states = { idle: 'Готов к запуску', starting: 'Запуск симулятора…', running: 'Проезд идёт', stopping: 'Останавливаем…', finished: 'Проезд завершён', stopped: 'Проезд остановлен', error: 'Ошибка запуска' };
    const statusText = app.online ? (!showRun && !isActive ? 'Готов к запуску' : states[state?.status] || 'Готов к запуску') : 'Нет связи';
    if ($('status').textContent !== statusText) $('status').textContent = statusText;
    $('status').className = `status ${state?.status || ''}`;
    const differs = isActive && selected.id !== state.map_id;
    $('browsing-note').hidden = !differs;
    $('browsing-note').textContent = `Выбрана следующая карта: ${selected.title || selected.name}. В эфире остаётся ${map.title || map.name}. Остановите текущий проезд, чтобы запустить выбранную карту.`;
    $('start').disabled = isActive || app.busy || !app.online;
    $('start').textContent = state?.map_id === selected.id && !isActive ? '↻ Повторить карту' : '▶ Запустить карту';
    $('stop').disabled = !isActive || state?.status === 'stopping' || app.busy || !app.online;
    $('speed').disabled = isActive || app.busy;
    $('speed-note').textContent = isActive ? `Текущий просмотр: ${state.speed}× · физический шаг неизменен` : 'Скорость задаётся перед запуском';
    $('known').textContent = outcome(selected.known_result).label;
    $('known-detail').textContent = resultDetail(selected.known_result);
    const currentMap = app.maps.find(m => m.id === state?.map_id);
    const report = state?.result;
    $('current').textContent = isActive ? 'Проезд ещё не завершён' : state?.status === 'stopped' ? 'Остановлено пользователем' : state?.status === 'error' ? 'Ошибка · результат не получен' : report ? outcome(report).label : 'Ещё не запускали';
    $('current-detail').textContent = currentMap ? `${currentMap.title || currentMap.name} · ${report ? resultDetail(report) : isActive ? 'Ждём итог проезда.' : 'Итоговый результат отсутствует.'}` : 'Запуск использует сохранённые веса, без обучения.';
    $('frame-label').textContent = frame ? `Шаг ${frame.step} · t = ${fmt(frame.time, 2, ' с')}` : 'Начальная позиция · проезд ещё не показан';
    $('time').textContent = fmt(frame?.time, 1, ' с'); $('distance').textContent = fmt(frame?.goal_distance, 2, ' м');
    $('velocity').textContent = fmt(frame?.speed, 2, ' м/с'); $('contacts').textContent = finite(frame?.contacts) ? frame.contacts : '—';
    $('throttle').textContent = fmt(frame?.action?.[0]); $('steering').textContent = fmt(frame?.action?.[1]); $('arrival').textContent = fmt(frame?.arrival_elapsed, 2, ' с');
    $('visible').textContent = frame ? (frame.target_visible === true ? 'Цель в кадре' : frame.target_visible === false ? 'Цель вне кадра' : 'Видимость неизвестна') : '—';
    const preview = typeof frame?.preview === 'string' && frame.preview ? frame.preview : null;
    if (preview !== lastPreview) {
      lastPreview = preview;
      if (preview) $('camera').src = preview.startsWith('data:image/jpeg;base64,') ? preview : `data:image/jpeg;base64,${preview}`;
      else $('camera').removeAttribute('src');
    }
    $('camera').hidden = !preview; $('camera-empty').hidden = !!preview;
    $('camera-empty').textContent = showRun && isActive ? 'Ждём первый кадр симулятора…' : 'Изображение появится после запуска';
    Array.from($('ranges').children).forEach((element, i) => {
      const value = frame?.ranges?.[i]; element.querySelector('output').textContent = fmt(value, 2);
      element.querySelector('.range-fill').style.height = `${finite(value) ? Math.max(0, Math.min(100, value / 1.2 * 100)) : 0}%`;
      element.title = finite(value) ? `Канал ${i + 1}: ${value.toFixed(3)} м. Шкала 0–1,2 м.` : `Канал ${i + 1}: нет данных`;
    });
    drawMap($('world'), map, frame, trail);
    if ($('activity-section').open) drawActivity(frame);
    if (state?.error) showError(state.error);
  }
  function acceptState(state) {
    if (!state || typeof state.status !== 'string') throw new Error('Некорректный ответ состояния симулятора.');
    const oldHistory = JSON.stringify(app.state?.history || {}), oldActive = active(), oldMap = app.state?.map_id;
    const wasOffline = !app.online;
    app.state = state; setConnection(true);
    if (wasOffline && app.error.startsWith('Не удалось получить данные:')) showError('');
    if (oldHistory !== JSON.stringify(state.history || {}) || oldActive !== active() || oldMap !== state.map_id) renderCards();
    renderView();
  }
  async function mutate(action, body) {
    if (app.busy) return;
    app.busy = true; showError(''); renderView();
    // Wait for any in-flight read before mutation, preventing a stale poll from replacing its response.
    try { if (pollPending) await pollPending; acceptState(await request(action, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })); }
    catch (error) { showError(error.message); }
    finally { app.busy = false; renderView(); }
  }
  async function loadCatalogue() {
    const data = await request('maps');
    if (!Array.isArray(data.maps) || !data.maps.length) throw new Error('Каталог карт пуст или недоступен.');
    app.maps = data.maps; app.arena = { ...app.arena, ...data.arena }; app.selected = app.maps[0];
    $('total').textContent = app.maps.length;
    $('model-label').textContent = data.model?.label || 'Сохранённая модель';
    $('model-detail').textContent = `${data.model?.neurons ?? '—'} нейронов · ${data.model?.edges ?? '—'} связей`;
    $('model-hash').textContent = data.model?.sha256 || 'Хеш не предоставлен';
    for (const group of new Set(app.maps.map(m => m.group))) { const option = document.createElement('option'); option.value = group; option.textContent = groupLabel(group); $('group').append(option); }
    app.loaded = true; renderCards(); renderView();
  }
  for (let i = 0; i < 6; i++) {
    const range = document.createElement('div'); range.className = 'range';
    const rangeNames = ['Вперёд', 'Диаг. Л', 'Диаг. П', 'Бок. Л', 'Бок. П', 'Назад'];
    range.innerHTML = `<span>${rangeNames[i]}</span><div class="range-track"><div class="range-fill"></div></div><output>—</output>`;
    $('ranges').append(range);
  }
  $('search').addEventListener('input', renderCards); $('group').addEventListener('change', renderCards); $('outcome').addEventListener('change', renderCards);
  $('start').addEventListener('click', () => { if (app.selected && !active()) mutate('start', { map_id: app.selected.id, speed: Number($('speed').value) }); });
  $('stop').addEventListener('click', () => mutate('stop', {}));
  $('activity-section').addEventListener('toggle', () => { if ($('activity-section').open) drawActivity(viewData().frame); });
  $('camera').addEventListener('error', () => { $('camera').hidden = true; $('camera-empty').hidden = false; $('camera-empty').textContent = 'Не удалось отобразить кадр камеры.'; });
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderCards(); renderView(); }, 100); });
  let pollPending = null;
  async function poll() {
    try {
      if (!app.busy) {
        pollPending = (async () => { if (!app.loaded) await loadCatalogue(); acceptState(await request('state')); })();
        await pollPending;
      }
    } catch (error) { setConnection(false); showError(`Не удалось получить данные: ${error.message} Повторное подключение выполняется автоматически.`); renderView(); }
    finally { pollPending = null; setTimeout(poll, app.online ? 200 : 1500); }
  }
  poll();
})();
