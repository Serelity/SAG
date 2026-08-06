"use strict";

const FIELDS = ["title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean"];
const MODES = ["problem", "question", "request", "suggestion", "praise", "historical_response", "current_stance"];
const TIMES = ["current", "historical"];
const LOCATION_TYPES = ["road", "intersection", "poi"];
const INTENTS = ["投诉", "举报", "求助", "咨询", "建议", "表扬", "催办", "反馈", "其他"];
const EMOTIONS = ["愤怒", "不满", "焦虑", "无奈", "悲伤", "感谢", "认可"];
const SATISFACTION = ["unknown", "satisfied", "dissatisfied", "mixed"];
const URGENCY = ["normal", "high", "critical"];
let record = null;
let summary = null;
let currentIndex = 0;
let dirty = false;
let selectedEvidence = null;

const $ = (id) => document.getElementById(id);
function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function select(values, value, className = "") {
  const node = element("select", className);
  for (const item of values) {
    const option = element("option", "", item);
    option.value = item;
    option.selected = item === value;
    node.append(option);
  }
  return node;
}
function input(value = "", className = "", placeholder = "") {
  const node = element("input", className);
  node.value = value || "";
  node.placeholder = placeholder;
  return node;
}
function markDirty() {
  dirty = true;
  $("progress").classList.add("dirty");
}
function refreshProgress() {
  const counts = summary.status_counts || {};
  $("progress").textContent = `${counts.completed || 0}/${summary.records} 已完成`;
  $("progress").classList.toggle("dirty", dirty);
}
function showValidation(message, ok = false) {
  const node = $("validation");
  node.textContent = message;
  node.className = ok ? "ok" : "error";
}
async function api(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const body = await response.json();
  if (!response.ok && response.status !== 422) throw new Error(body.error || `HTTP ${response.status}`);
  return {status: response.status, body};
}
function fillFieldSelect(node, value) {
  for (const field of FIELDS) {
    const option = element("option", "", field);
    option.value = field;
    option.selected = field === value;
    node.append(option);
  }
}
function captureSelection() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) return;
  const anchor = selection.anchorNode && (selection.anchorNode.nodeType === 1 ? selection.anchorNode : selection.anchorNode.parentElement);
  const focus = selection.focusNode && (selection.focusNode.nodeType === 1 ? selection.focusNode : selection.focusNode.parentElement);
  const left = anchor && anchor.closest && anchor.closest(".source-text");
  const right = focus && focus.closest && focus.closest(".source-text");
  const text = selection.toString();
  if (left && left === right && text) selectedEvidence = {field: left.dataset.field, evidence: text};
}
document.addEventListener("selectionchange", captureSelection);
function selectionButton(row) {
  const button = element("button", "selection", "取选中文字");
  button.type = "button";
  button.addEventListener("mousedown", (event) => event.preventDefault());
  button.addEventListener("click", () => {
    if (!selectedEvidence) {
      showValidation("请先在上方某一个原文字段内选中文字。");
      return;
    }
    row.querySelector(".field").value = selectedEvidence.field;
    row.querySelector(".evidence").value = selectedEvidence.evidence;
    const surface = row.querySelector(".surface");
    if (surface && !surface.value) surface.value = selectedEvidence.evidence;
    markDirty();
  });
  return button;
}
function removeButton(row) {
  const button = element("button", "remove danger", "删除");
  button.type = "button";
  button.addEventListener("click", () => { row.remove(); markDirty(); });
  return button;
}
function memberRow(value = {}, location = false) {
  const row = element("div", location ? "member-row location-row" : "member-row");
  if (location) row.append(select(LOCATION_TYPES, value.type || "road", "location-type"));
  row.append(input(value.surface, "surface", "surface"));
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  row.append(field, input(value.evidence, "evidence wide", "逐字 evidence"), selectionButton(row), removeButton(row));
  return row;
}
function memberGroup(title, key, values, location = false) {
  const group = element("div", "member-group");
  group.dataset.key = key;
  const head = element("div", "member-group-title");
  head.append(element("strong", "", title));
  const add = element("button", "", "添加");
  add.type = "button";
  add.addEventListener("click", () => { group.append(memberRow({}, location)); markDirty(); });
  head.append(add);
  group.append(head);
  for (const value of values || []) group.append(memberRow(value, location));
  return group;
}
function issueCard(value = {}) {
  const card = element("article", "issue-card");
  const head = element("div", "issue-head");
  head.append(element("h3", "", "Issue"));
  const controls = element("div");
  controls.append(select(MODES, value.mode || "problem", "mode"), select(TIMES, value.time_scope || "current", "time-scope"));
  const remove = element("button", "danger", "删除 issue");
  remove.type = "button";
  remove.addEventListener("click", () => { card.remove(); markDirty(); });
  controls.append(remove);
  head.append(controls);
  card.append(
    head,
    memberGroup("对象 objects", "objects", value.objects),
    memberGroup("谓词 predicates", "predicates", value.predicates),
    memberGroup("诉求动作 actions", "actions", value.actions),
    memberGroup("地点 locations", "locations", value.locations, true),
  );
  return card;
}
function discourseRow(value, labels, emotion = false) {
  const row = element("div", emotion ? "discourse-row emotion-row" : "discourse-row");
  row.append(select(labels, value.label || labels[0], "label"));
  if (emotion) row.append(select(["1", "2", "3"], String(value.intensity || 1), "intensity"));
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  row.append(field, input(value.evidence, "evidence", "逐字 evidence"), selectionButton(row), removeButton(row));
  return row;
}
function groundedFields(container, value, includeTarget = false) {
  if (includeTarget) container.append(input(value.target, "target", "评价 target"));
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  const evidence = input(value.evidence, "evidence", "逐字 evidence");
  const row = element("div", "discourse-row");
  row.append(field, evidence, selectionButton(row));
  container.append(row);
}
function render() {
  dirty = false;
  selectedEvidence = null;
  $("record-index").value = record.index + 1;
  $("record-total").textContent = `/ ${record.records}`;
  $("subset").textContent = record.subset;
  $("previous").disabled = record.index === 0;
  $("next").disabled = record.index + 1 === record.records;
  const fields = $("clean-fields"); fields.replaceChildren();
  for (const field of FIELDS) {
    const wrapper = element("div", "source-field");
    wrapper.append(element("div", "source-label", field));
    const text = element("div", "source-text", record.clean_fields[field] || "（空）");
    text.dataset.field = field;
    wrapper.append(text); fields.append(wrapper);
  }
  const metadata = $("metadata"); metadata.replaceChildren();
  for (const [key, value] of Object.entries(record.metadata || {})) {
    metadata.append(element("dt", "", key), element("dd", "", String(value)));
  }
  const issues = $("issues"); issues.replaceChildren();
  for (const issue of record.issues || []) issues.append(issueCard(issue));
  const intents = $("intents"); intents.replaceChildren();
  for (const value of record.declared_intents || []) intents.append(discourseRow(value, INTENTS));
  const emotions = $("emotions"); emotions.replaceChildren();
  for (const value of record.direct_emotions || []) emotions.append(discourseRow(value, EMOTIONS, true));
  const satisfaction = $("satisfaction"); satisfaction.replaceChildren(element("legend", "", "满意度"));
  satisfaction.append(select(SATISFACTION, record.satisfaction.label || "unknown", "satisfaction-label"));
  groundedFields(satisfaction, record.satisfaction, true);
  const urgency = $("urgency"); urgency.replaceChildren(element("legend", "", "紧迫度"));
  urgency.append(select(URGENCY, record.urgency.level || "normal", "urgency-level"));
  groundedFields(urgency, record.urgency);
  $("notes").value = record.annotation.notes || "";
  showValidation(record.annotation.status === "completed" ? "本条当前已完成；修改后请重新保存。" : "", true);
  refreshProgress();
}
function readMember(row, location = false) {
  const result = {
    surface: row.querySelector(".surface").value.trim(),
    field: row.querySelector(".field").value,
    evidence: row.querySelector(".evidence").value,
  };
  if (location) result.type = row.querySelector(".location-type").value;
  return result;
}
function readIssues() {
  return [...$("issues").querySelectorAll(":scope > .issue-card")].map((card) => {
    const issue = {mode: card.querySelector(".mode").value, time_scope: card.querySelector(".time-scope").value};
    for (const group of card.querySelectorAll(".member-group")) {
      issue[group.dataset.key] = [...group.querySelectorAll(":scope > .member-row")].map((row) => readMember(row, group.dataset.key === "locations"));
    }
    return issue;
  });
}
function readDiscourse(container, emotion = false) {
  return [...container.querySelectorAll(":scope > .discourse-row")].map((row) => {
    const value = {label: row.querySelector(".label").value, field: row.querySelector(".field").value, evidence: row.querySelector(".evidence").value};
    if (emotion) value.intensity = Number(row.querySelector(".intensity").value);
    return value;
  });
}
function readSatisfaction() {
  const fieldset = $("satisfaction");
  const label = fieldset.querySelector(".satisfaction-label").value;
  if (label === "unknown") return {label, target: "", evidence: ""};
  return {label, target: fieldset.querySelector(".target").value.trim(), field: fieldset.querySelector(".field").value, evidence: fieldset.querySelector(".evidence").value};
}
function readUrgency() {
  const fieldset = $("urgency");
  const level = fieldset.querySelector(".urgency-level").value;
  if (level === "normal") return {level, evidence: ""};
  return {level, field: fieldset.querySelector(".field").value, evidence: fieldset.querySelector(".evidence").value};
}
function payload(status) {
  return {
    issues: readIssues(),
    declared_intents: readDiscourse($("intents")),
    direct_emotions: readDiscourse($("emotions"), true),
    satisfaction: readSatisfaction(),
    urgency: readUrgency(),
    status,
    notes: $("notes").value,
  };
}
async function save(status) {
  try {
    const result = await api(`/api/records/${record.index}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({revision: record.revision, payload: payload(status)}),
    });
    if (!result.body.saved) {
      showValidation("未保存：\n" + (result.body.validation.errors || []).join("\n"));
      return false;
    }
    record.revision = result.body.revision;
    record.annotation.status = status;
    summary.status_counts = result.body.status_counts;
    dirty = false;
    refreshProgress();
    const warnings = result.body.validation.warnings || [];
    showValidation(`保存成功${warnings.length ? "；警告：" + warnings.join(", ") : ""}`, true);
    return true;
  } catch (error) {
    showValidation("保存失败：" + error.message);
    return false;
  }
}
async function load(index) {
  if (dirty && !window.confirm("当前修改尚未保存，确定离开吗？")) return;
  try {
    const result = await api(`/api/records/${index}`);
    record = result.body; currentIndex = index; render();
  } catch (error) { showValidation("加载失败：" + error.message); }
}
$("add-issue").addEventListener("click", () => { $("issues").append(issueCard()); markDirty(); });
$("add-intent").addEventListener("click", () => { $("intents").append(discourseRow({}, INTENTS)); markDirty(); });
$("add-emotion").addEventListener("click", () => { $("emotions").append(discourseRow({}, EMOTIONS, true)); markDirty(); });
$("previous").addEventListener("click", () => load(currentIndex - 1));
$("next").addEventListener("click", () => load(currentIndex + 1));
$("go").addEventListener("click", () => load(Number($("record-index").value) - 1));
$("save-draft").addEventListener("click", () => save("in_progress"));
$("save-complete").addEventListener("click", () => save("completed"));
document.querySelector("main").addEventListener("input", markDirty);
document.querySelector("main").addEventListener("change", markDirty);
window.addEventListener("beforeunload", (event) => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); save("in_progress"); }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); save("completed"); }
});

(async () => {
  try {
    summary = (await api("/api/summary")).body;
    await load(0);
  } catch (error) { showValidation("初始化失败：" + error.message); }
})();
