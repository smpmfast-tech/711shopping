const checkoutItems = document.querySelector("#checkout-items");
const checkoutEmpty = document.querySelector("#checkout-empty");
const checkoutArea = document.querySelector("#checkout-area");
const orderForm = document.querySelector("#order-form");
const orderState = document.querySelector("#order-state");
const orderDone = document.querySelector("#order-done");
const basketCount = document.querySelector("#basket-count");
const BASKET_KEY = "godelivery-basket";
const productById = new Map();
let basket = new Map();
let privacyPolicyConfigured = false;
let contentLoaded = false;

function showState(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("form-state-error", isError);
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function getPrice(product) {
  return product.sale_price ?? product.regular_price;
}

function formatPrice(value) {
  return new Intl.NumberFormat("th-TH", {
    style: "currency",
    currency: "THB",
    maximumFractionDigits: 2,
  }).format(value);
}

function renderSummary() {
  let totalItems = 0;
  let totalPrice = 0;
  const lines = [];
  let hasUnavailableProduct = false;
  for (const [productId, quantity] of basket.entries()) {
    totalItems += quantity;
    const product = productById.get(productId);
    const line = makeElement("div", "basket-line");
    const description = document.createElement("div");
    if (!product) {
      hasUnavailableProduct = true;
      description.append(makeElement("strong", "", `สินค้า ${productId} ไม่พร้อมจำหน่าย`));
      description.append(makeElement("small", "", "กรุณากลับไปนำรายการออกจากตะกร้า"));
    } else {
      description.append(makeElement("strong", "", product.name));
      if (quantity > product.stock_quantity) {
        hasUnavailableProduct = true;
        description.append(makeElement("small", "", `จำนวนเกินสต็อกที่เหลือ ${product.stock_quantity} ชิ้น`));
      }
      const linePrice = getPrice(product) * quantity;
      totalPrice += linePrice;
      description.append(makeElement("small", "", `${formatPrice(getPrice(product))} × ${quantity} = ${formatPrice(linePrice)}`));
    }
    line.append(description);
    lines.push(line);
  }
  checkoutItems.replaceChildren(...lines);
  checkoutEmpty.hidden = basket.size > 0;
  checkoutArea.hidden = basket.size === 0;
  basketCount.textContent = String(totalItems);
  document.querySelector("#checkout-total").textContent = formatPrice(totalPrice);
  orderForm.querySelector('button[type="submit"]').disabled =
    hasUnavailableProduct || !privacyPolicyConfigured;
}

async function loadSiteContent() {
  try {
    const response = await fetch("/api/v1/site/content", { cache: "no-store" });
    if (!response.ok) throw new Error("โหลดข้อมูลสินค้าไม่สำเร็จ กรุณาลองใหม่ภายหลัง");
    const content = await response.json();
    productById.clear();
    for (const product of content.products || []) {
      if (product.active) productById.set(product.product_id, product);
    }
    privacyPolicyConfigured = Boolean(content.security?.body?.trim());
    orderForm.querySelector('input[name="privacy_consent"]').disabled = !privacyPolicyConfigured;
    contentLoaded = true;
  } catch (error) {
    console.error("Could not load the product catalog for checkout.", error);
    showState(orderState, "โหลดข้อมูลสินค้าไม่สำเร็จ กรุณารีเฟรชหน้าอีกครั้ง", true);
  }
  renderSummary();
}

async function prefillMemberProfile() {
  try {
    const response = await fetch("/api/v1/member/me", { credentials: "same-origin" });
    if (!response.ok) return;
    const me = await response.json();
    if (!me.authenticated || me.stage !== "active" || !me.profile) return;
    const map = { customer: "name", address: "address", address1: "address1", postcode: "postcode", phone_number: "phone" };
    for (const [inputName, key] of Object.entries(map)) {
      const input = orderForm.elements[inputName];
      if (input && !input.value && me.profile[key]) input.value = me.profile[key];
    }
  } catch (error) {
    // Prefill is best effort; the form stays editable without it.
  }
}

try {
  const savedBasket = JSON.parse(localStorage.getItem(BASKET_KEY) || "[]");
  if (Array.isArray(savedBasket)) {
    basket = new Map(savedBasket.filter((entry) =>
      Array.isArray(entry) && typeof entry[0] === "string" &&
      Number.isInteger(entry[1]) && entry[1] > 0
    ));
  }
} catch (error) {
  console.error("Could not restore the saved shopping basket.", error);
}

orderForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton = orderForm.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  showState(orderState, "กำลังส่งคำสั่งซื้อ…");
  const formData = new FormData(orderForm);
  const payload = {
    customer: String(formData.get("customer") || "").trim(),
    address: String(formData.get("address") || "").trim(),
    address1: String(formData.get("address1") || "").trim(),
    postcode: String(formData.get("postcode") || "").trim(),
    phone_number: String(formData.get("phone_number") || "").trim(),
    remark: String(formData.get("remark") || "").trim(),
    privacy_consent: formData.get("privacy_consent") === "on",
    items: Array.from(basket, ([product_id, quantity]) => ({ product_id, quantity })),
  };
  try {
    const response = await fetch("/api/v1/site/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || "ส่งคำสั่งซื้อไม่สำเร็จ");
    basket.clear();
    try {
      localStorage.setItem(BASKET_KEY, "[]");
    } catch (error) {
      console.error("Could not clear the shopping basket.", error);
    }
    orderForm.reset();
    checkoutArea.hidden = true;
    showState(orderState, "");
    showState(orderDone, `${result.message} เลขที่รายการ ${result.ordernumber}`);
  } catch (error) {
    showState(orderState, error.message, true);
    renderSummary();
  }
});

renderSummary();
loadSiteContent();
prefillMemberProfile();
