"use strict";

const FIELDS = ["title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean"];
const FIELD_LABELS = {
  title_clean: "工单标题",
  case_content_clean: "工单内容",
  case_goal_clean: "诉求目标",
  address_detail_clean: "详细地址",
};
const MODES = [
  {value: "problem", label: "问题/故障（发生了什么问题）"},
  {value: "question", label: "咨询/提问（想了解什么）"},
  {value: "request", label: "请求办理（希望采取什么动作）"},
  {value: "suggestion", label: "意见建议"},
  {value: "praise", label: "表扬"},
  {value: "historical_response", label: "历史办理/答复"},
  {value: "current_stance", label: "对历史办理的当前态度"},
];
const TIMES = [
  {value: "current", label: "当前事实/诉求"},
  {value: "historical", label: "历史情况/历史答复"},
];
const LOCATION_TYPES = [
  {value: "road", label: "道路"},
  {value: "intersection", label: "道路交叉口"},
  {value: "poi", label: "具体地点/设施（POI）"},
];
const INTENTS = ["投诉", "举报", "求助", "咨询", "建议", "表扬", "催办", "反馈", "其他"];
const EMOTIONS = ["愤怒", "不满", "焦虑", "无奈", "悲伤", "感谢", "认可"];
const SATISFACTION = [
  {value: "unknown", label: "未知/未直接表达"},
  {value: "satisfied", label: "满意"},
  {value: "dissatisfied", label: "不满意"},
  {value: "mixed", label: "褒贬并存"},
];
const URGENCY = [
  {value: "normal", label: "普通（无直接紧迫表达）"},
  {value: "high", label: "较紧迫（明确催促/久拖未决）"},
  {value: "critical", label: "危急（人身/生命/重大安全风险）"},
];
const SUBSET_LABELS = {production: "生产分布样本", challenge: "挑战样本"};
const METADATA_LABELS = {
  service_object_type: "系统登记诉求类型",
  area: "行政区域",
  street: "街道",
  community: "社区",
  type1: "系统一级分类",
  type2: "系统二级分类",
  type3: "系统三级分类",
  event_time: "工单时间",
};
const MODE_HELP = {
  problem: "填写已经发生的异常、故障、阻碍或负面状态。通常至少填写“对象”和“状态/问题”。",
  question: "填写正常咨询或查询。把咨询对象填在“对象”，具体想问的内容填在“状态/问题”。不要把“询问”标成故障。",
  request: "填写希望部门采取的处理动作。通常填写“对象”和“诉求动作”；动作不是已经发生的问题。",
  suggestion: "填写市民提出的改进意见。通常把相关事项填在“对象”，具体建议填在“诉求动作”。",
  praise: "填写对人员、部门或服务的明确表扬，只标原文直接表达的内容。",
  historical_response: "只填写过去部门答复、办理过程或处理结论；时间范围必须是“历史”。",
  current_stance: "填写市民现在对历史办理的态度，例如不认可、仍未解决、再次反映；时间范围必须是“当前”。",
};
const VALIDATION_LABELS = {
  completed_annotation_without_issue: "完成标注时至少要有一个事实/诉求单元。",
  empty_issue: "存在一个完全空的单元：请填写内容或删除该单元。",
  issue_invalid_mode: "单元类型无效，请重新选择。",
  issue_invalid_time_scope: "时间范围无效，请重新选择。",
  historical_response_not_historical: "“历史办理/答复”的时间范围必须选择“历史”。",
  current_stance_not_current: "“对历史办理的当前态度”的时间范围必须选择“当前”。",
  duplicate_issue: "存在两个内容完全相同的单元，请合并或删除重复项。",
  duplicate_issue_member: "同一单元中存在重复成员。",
  duplicate_issue_location: "同一单元中存在重复地点。",
  issue_member_missing_surface: "对象、状态或动作的“标准短语”不能为空。",
  issue_member_missing_evidence: "对象、状态或动作缺少逐字证据。",
  issue_member_missing_field: "请选择证据所在的原文字段。",
  issue_member_invalid_field: "证据字段无效。",
  issue_member_evidence_not_in_field: "证据与所选原文字段不完全一致；请从上方原文重新选择。",
  issue_member_evidence_not_in_clean_fields: "证据没有出现在任何脱敏原文字段中。",
  issue_member_surface_not_in_evidence: "标准短语必须完整出现在逐字证据中。",
  issue_location_missing_surface: "地点的标准短语不能为空。",
  issue_location_missing_evidence: "地点缺少逐字证据。",
  issue_location_missing_field: "请选择地点证据所在的原文字段。",
  issue_location_invalid_field: "地点证据字段无效。",
  issue_location_evidence_not_in_field: "地点证据与所选原文字段不完全一致。",
  issue_location_surface_not_in_evidence: "地点标准短语必须完整出现在逐字证据中。",
  issue_location_invalid_type: "地点类型无效。",
  intent_missing_evidence: "意图缺少逐字证据。",
  intent_missing_field: "请选择意图证据所在的原文字段。",
  intent_evidence_not_in_field: "意图证据与所选原文字段不完全一致。",
  intent_evidence_not_in_clean_fields: "意图证据没有出现在脱敏原文中。",
  intent_invalid_label: "意图类别无效。",
  emotion_missing_evidence: "情绪缺少逐字证据。",
  emotion_missing_field: "请选择情绪证据所在的原文字段。",
  emotion_evidence_not_in_field: "情绪证据与所选原文字段不完全一致。",
  emotion_evidence_not_in_clean_fields: "情绪证据没有出现在脱敏原文中。",
  emotion_invalid_label: "情绪类别无效。",
  emotion_invalid_intensity: "情绪强度无效。",
  satisfaction_missing_target: "选择了非“未知”的满意度后，必须填写评价对象。",
  satisfaction_missing_evidence: "满意度缺少原文直接证据。",
  satisfaction_missing_field: "请选择满意度证据所在的原文字段。",
  satisfaction_evidence_not_in_field: "满意度证据与所选原文字段不完全一致。",
  satisfaction_evidence_not_in_clean_fields: "满意度证据没有出现在脱敏原文中。",
  unknown_satisfaction_has_grounding: "满意度为“未知”时不应填写评价对象或证据。",
  urgency_missing_evidence: "较紧迫或危急必须有原文直接证据。",
  urgency_missing_field: "请选择紧迫度证据所在的原文字段。",
  urgency_evidence_not_in_field: "紧迫度证据与所选原文字段不完全一致。",
  urgency_evidence_not_in_clean_fields: "紧迫度证据没有出现在脱敏原文中。",
  normal_urgency_has_evidence: "紧迫度为“普通”时不应填写紧迫证据。",
  annotation_payload_invalid: "页面提交内容不完整或含有不允许修改的字段，请刷新后重试。",
  annotation_revision_conflict: "该文件已被其他页面修改。请刷新后重新填写当前条。",
  annotation_file_changed_externally: "标注文件被页面外的程序修改，已拒绝覆盖。请停止后检查文件。",
};
const WARNING_LABELS = {
  annotation_in_progress: "当前已保存为草稿，尚未标记完成。",
  problem_without_predicate: "问题/故障单元没有填写状态或问题，请确认是否遗漏。",
  question_without_predicate: "咨询/提问单元没有填写具体问题，请确认是否遗漏。",
  historical_response_without_predicate: "历史答复单元没有填写办理结论，请确认是否遗漏。",
  current_stance_without_predicate: "当前态度单元没有填写态度或未解决状态，请确认是否遗漏。",
  request_without_action: "请求办理单元没有填写诉求动作，请确认是否遗漏。",
  suggestion_without_action: "意见建议单元没有填写具体建议，请确认是否遗漏。",
};
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
function select(values, value, className = "", ariaLabel = "") {
  const node = element("select", className);
  if (ariaLabel) node.setAttribute("aria-label", ariaLabel);
  for (const item of values) {
    const optionValue = typeof item === "string" ? item : item.value;
    const optionLabel = typeof item === "string" ? item : item.label;
    const option = element("option", "", optionLabel);
    option.value = optionValue;
    option.selected = optionValue === value;
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
function validationText(code, warning = false) {
  const labels = warning ? WARNING_LABELS : VALIDATION_LABELS;
  return labels[code] || `校验提示（${code}）`;
}
function validationLines(codes, warning = false) {
  return (codes || []).map((code, index) => `${index + 1}. ${validationText(code, warning)}`);
}
function renumberIssues() {
  [...$("issues").querySelectorAll(":scope > .issue-card")].forEach((card, index) => {
    card.querySelector("h3").textContent = `事实/诉求单元 ${index + 1}`;
  });
}
async function api(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const body = await response.json();
  if (!response.ok && response.status !== 422) throw new Error(body.error || `HTTP ${response.status}`);
  return {status: response.status, body};
}
function fillFieldSelect(node, value) {
  node.setAttribute("aria-label", "证据所在原文字段");
  for (const field of FIELDS) {
    const option = element("option", "", FIELD_LABELS[field]);
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
  const button = element("button", "selection", "使用选中文字");
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
  if (location) row.append(select(LOCATION_TYPES, value.type || "road", "location-type", "地点类型"));
  row.append(input(value.surface, "surface", "标准短语，例如：路灯/不亮/维修"));
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  row.append(field, input(value.evidence, "evidence wide", "原文中的连续逐字证据"), selectionButton(row), removeButton(row));
  return row;
}
function memberGroup(title, key, values, location = false) {
  const group = element("div", "member-group");
  group.dataset.key = key;
  const head = element("div", "member-group-title");
  head.append(element("strong", "", title));
  const add = element("button", "", "添加一项");
  add.type = "button";
  add.addEventListener("click", () => { group.append(memberRow({}, location)); markDirty(); });
  head.append(add);
  group.append(head);
  group.append(element("div", "column-help", location
    ? "每行依次填写：地点类型｜地点短语｜证据所在字段｜原文逐字证据"
    : "每行依次填写：标准短语｜证据所在字段｜原文逐字证据"));
  for (const value of values || []) group.append(memberRow(value, location));
  return group;
}
function issueCard(value = {}) {
  const card = element("article", "issue-card");
  const head = element("div", "issue-head");
  head.append(element("h3", "", "事实/诉求单元"));
  const controls = element("div");
  const mode = select(MODES, value.mode || "problem", "mode", "单元类型");
  const timeScope = select(TIMES, value.time_scope || "current", "time-scope", "时间范围");
  controls.append(mode, timeScope);
  const remove = element("button", "danger", "删除本单元");
  remove.type = "button";
  remove.addEventListener("click", () => { card.remove(); renumberIssues(); markDirty(); });
  controls.append(remove);
  head.append(controls);
  const modeHelp = element("p", "mode-help");
  const updateModeHelp = () => { modeHelp.textContent = MODE_HELP[mode.value] || ""; };
  mode.addEventListener("change", updateModeHelp);
  updateModeHelp();
  card.append(
    head,
    modeHelp,
    memberGroup("对象/事项（什么东西或什么事）", "objects", value.objects),
    memberGroup("状态/问题（怎么了、问什么、当前态度）", "predicates", value.predicates),
    memberGroup("诉求动作（希望采取什么处理）", "actions", value.actions),
    memberGroup("相关地点（没有明确专名就留空）", "locations", value.locations, true),
  );
  return card;
}
function discourseRow(value, labels, emotion = false) {
  const row = element("div", emotion ? "discourse-row emotion-row" : "discourse-row");
  row.append(select(labels, value.label || labels[0], "label", emotion ? "情绪类别" : "意图类别"));
  if (emotion) row.append(select([
    {value: "1", label: "轻微"}, {value: "2", label: "明显"}, {value: "3", label: "强烈"},
  ], String(value.intensity || 1), "intensity", "情绪强度"));
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  row.append(field, input(value.evidence, "evidence", "原文中的连续逐字证据"), selectionButton(row), removeButton(row));
  return row;
}
function groundedFields(container, value, includeTarget = false) {
  if (includeTarget) {
    const target = input(value.target, "target grounding", "评价对象，例如：办理结果/工作人员服务");
    container.append(target);
  }
  const field = element("select", "field");
  fillFieldSelect(field, value.field || "case_content_clean");
  const evidence = input(value.evidence, "evidence", "原文中的连续逐字证据");
  const row = element("div", "discourse-row grounding");
  row.append(field, evidence, selectionButton(row));
  container.append(row);
}
function render() {
  dirty = false;
  selectedEvidence = null;
  $("record-index").value = record.index + 1;
  $("record-total").textContent = `/ ${record.records}`;
  $("subset").textContent = SUBSET_LABELS[record.subset] || "抽样记录";
  $("previous").disabled = record.index === 0;
  $("next").disabled = record.index + 1 === record.records;
  const fields = $("clean-fields"); fields.replaceChildren();
  for (const field of FIELDS) {
    const wrapper = element("div", "source-field");
    wrapper.append(element("div", "source-label", FIELD_LABELS[field]));
    const text = element("div", "source-text", record.clean_fields[field] || "（空）");
    text.dataset.field = field;
    wrapper.append(text); fields.append(wrapper);
  }
  const metadata = $("metadata"); metadata.replaceChildren();
  for (const [key, value] of Object.entries(record.metadata || {})) {
    metadata.append(element("dt", "", METADATA_LABELS[key] || key), element("dd", "", String(value)));
  }
  const issues = $("issues"); issues.replaceChildren();
  for (const issue of record.issues || []) issues.append(issueCard(issue));
  renumberIssues();
  const intents = $("intents"); intents.replaceChildren();
  for (const value of record.declared_intents || []) intents.append(discourseRow(value, INTENTS));
  const emotions = $("emotions"); emotions.replaceChildren();
  for (const value of record.direct_emotions || []) emotions.append(discourseRow(value, EMOTIONS, true));
  const satisfaction = $("satisfaction"); satisfaction.replaceChildren(element("legend", "", "五、对办理结果的满意度"));
  satisfaction.append(element("p", "hint", "只标市民对办理结果或服务的直接评价；没有明确评价就保持“未知”。"));
  const satisfactionSelect = select(SATISFACTION, record.satisfaction.label || "unknown", "satisfaction-label", "满意度");
  satisfaction.append(satisfactionSelect);
  groundedFields(satisfaction, record.satisfaction, true);
  const updateSatisfaction = () => satisfaction.querySelectorAll(".grounding").forEach((node) => node.classList.toggle("hidden", satisfactionSelect.value === "unknown"));
  satisfactionSelect.addEventListener("change", updateSatisfaction); updateSatisfaction();
  const urgency = $("urgency"); urgency.replaceChildren(element("legend", "", "六、紧迫程度"));
  urgency.append(element("p", "hint", "只根据当前风险或明确催促判断；事情重要不等于危急。没有直接证据就保持“普通”。"));
  const urgencySelect = select(URGENCY, record.urgency.level || "normal", "urgency-level", "紧迫程度");
  urgency.append(urgencySelect);
  groundedFields(urgency, record.urgency);
  const updateUrgency = () => urgency.querySelectorAll(".grounding").forEach((node) => node.classList.toggle("hidden", urgencySelect.value === "normal"));
  urgencySelect.addEventListener("change", updateUrgency); updateUrgency();
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
      showValidation("未保存，请按下面提示修改：\n" + validationLines(result.body.validation.errors).join("\n"));
      return false;
    }
    record.revision = result.body.revision;
    record.annotation.status = status;
    summary.status_counts = result.body.status_counts;
    dirty = false;
    refreshProgress();
    const warnings = result.body.validation.warnings || [];
    showValidation(warnings.length ? `保存成功，但请确认以下提示：\n${validationLines(warnings, true).join("\n")}` : "保存成功。", true);
    return true;
  } catch (error) {
    showValidation("保存失败：" + validationText(error.message));
    return false;
  }
}
async function load(index) {
  if (dirty && !window.confirm("当前修改尚未保存，确定离开吗？")) return;
  try {
    const result = await api(`/api/records/${index}`);
    record = result.body; currentIndex = index; render();
  } catch (error) { showValidation("加载失败：" + validationText(error.message)); }
}
$("add-issue").addEventListener("click", () => { $("issues").append(issueCard()); renumberIssues(); markDirty(); });
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
