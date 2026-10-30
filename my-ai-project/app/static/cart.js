const basketItems = document.querySelector("#basket-items");
const basketEmpty = document.querySelector("#basket-empty");
const orderArea = document.querySelector("#order-area");
const checkoutButton = document.querySelector("#checkout-button");
const basketCount = document.querySelector("#basket-count");
const BASKET_KEY = "godelivery-basket";
const productById = new Map();
let basket = new Map();
let contentLoaded = false;

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

function persistBasket() {
  try {
    localStorage.setItem(BASKET_KEY, JSON.stringify(Array.from(basket.entries())));
  } catch (error) {
    console.error("Could not save the shopping basket on this device.", error);
  }
}

function renderBasket() {
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
      // Until the catalog loads, a missing product is not yet "unavailable".
      if (contentLoaded) {
        hasUnavailableProduct = true;
        description.append(makeElement("strong", "", `สินค้า ${productId} ไม่พร้อมจำหน่าย`));
        description.append(makeElement("small", "", "นำรายการออกจากตะกร้าก่อนดำเนินการสั่งซื้อ"));
      } else {
        description.append(makeElement("strong", "", `สินค้า ${productId}`));
        description.append(makeElement("small", "", "กำลังโหลดข้อมูลสินค้า…"));
        hasUnavailableProduct = true;
      }
    } else {
      const name = makeElement("a", "", product.name);
      name.href = `/products/${encodeURIComponent(productId)}`;
      const title = makeElement("strong");
      title.append(name);
      description.append(title);
      if (quantity > product.stock_quantity) {
        hasUnavailableProduct = true;
        description.append(makeElement("small", "", `จำนวนเกินสต็อกที่เหลือ ${product.stock_quantity} ชิ้น`));
      }
      const linePrice = getPrice(product) * quantity;
      totalPrice += linePrice;
      description.append(makeElement("small", "", `${formatPrice(getPrice(product))} × ${quantity} = ${formatPrice(linePrice)}`));
    }
    line.append(description);

    const decrease = makeElement("button", "", "−");
    decrease.type = "button";
    decrease.setAttribute("aria-label", `ลดจำนวน ${product?.name || productId}`);
    decrease.addEventListener("click", () => {
      if (quantity <= 1) basket.delete(productId);
      else basket.set(productId, quantity - 1);
      persistBasket();
      renderBasket();
    });
    const quantityLabel = makeElement("span", "", String(quantity));
    const increase = makeElement("button", "", "+");
    increase.type = "button";
    increase.setAttribute("aria-label", `เพิ่มจำนวน ${product?.name || productId}`);
    increase.disabled = !product || quantity >= product.stock_quantity;
    increase.addEventListener("click", () => {
      basket.set(productId, quantity + 1);
      persistBasket();
      renderBasket();
    });
    const remove = makeElement("button", "remove-item", "นำออก");
    remove.type = "button";
    remove.addEventListener("click", () => {
      basket.delete(productId);
      persistBasket();
      renderBasket();
    });
    line.append(decrease, quantityLabel, increase, remove);
    lines.push(line);
  }
  basketItems.replaceChildren(...lines);
  basketEmpty.hidden = basket.size > 0;
  orderArea.hidden = basket.size === 0;
  basketCount.textContent = String(totalItems);
  document.querySelector("#basket-total").textContent = formatPrice(totalPrice);
  checkoutButton.disabled = hasUnavailableProduct;
}

checkoutButton.addEventListener("click", () => {
  if (basket.size === 0) return;
  const day = String(new Date().getDate()).padStart(2, "0");
  const random = String(Math.floor(Math.random() * 1000)).padStart(3, "0");
  window.location.assign(`/checkout${day}${random}`);
});

async function loadSiteContent() {
  try {
    const response = await fetch("/api/v1/site/content", { cache: "no-store" });
    if (!response.ok) throw new Error("โหลดข้อมูลสินค้าไม่สำเร็จ กรุณาลองใหม่ภายหลัง");
    const content = await response.json();
    productById.clear();
    for (const product of content.products || []) {
      if (product.active) productById.set(product.product_id, product);
    }
    contentLoaded = true;
  } catch (error) {
    console.error("Could not load the product catalog for the basket.", error);
  }
  renderBasket();
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

window.addEventListener("storage", (event) => {
  if (event.key !== BASKET_KEY) return;
  try {
    basket = new Map(JSON.parse(event.newValue || "[]"));
  } catch (error) {
    return;
  }
  renderBasket();
});

renderBasket();
loadSiteContent();
