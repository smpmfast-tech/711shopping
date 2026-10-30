import re
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.admin import router as admin_router
from app.api.v1.endpoints.member import read_session as read_member_session
from app.api.v1.router import api_router
from app.config import settings

app = FastAPI(title=settings.APP_NAME, version="1.0.0")
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(admin_router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def prevent_search_indexing(request, call_next):
    response = await call_next(request)
    response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    return response


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt() -> PlainTextResponse:
    return PlainTextResponse(
        "User-agent: *\n"
        "Allow: /\n"
        "# Pages send noindex headers. Crawlers must be able to access them "
        "to process the noindex directive.\n"
    )


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "home.html")


@app.get("/product", include_in_schema=False)
async def product_catalog() -> FileResponse:
    return FileResponse(STATIC_DIR / "product.html")


@app.get("/cart", include_in_schema=False)
async def cart_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "cart.html")


CHECKOUT_SUFFIX_PATTERN = re.compile(r"(0[1-9]|[12][0-9]|3[01])[0-9]{3}")


@app.get("/checkout{suffix}", include_in_schema=False)
async def checkout_page(suffix: str) -> Response:
    if not CHECKOUT_SUFFIX_PATTERN.fullmatch(suffix):
        return RedirectResponse("/cart", status_code=302)
    return FileResponse(STATIC_DIR / "checkout.html", headers={"Cache-Control": "no-store"})


@app.get("/faq", include_in_schema=False)
async def faq_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "faq.html")


@app.get("/fag", include_in_schema=False)
async def faq_page_alias() -> RedirectResponse:
    return RedirectResponse("/faq", status_code=301)


@app.get("/products/{product_id:path}", include_in_schema=False)
async def product_detail(product_id: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "product-detail.html")


@app.get("/chatandshop", include_in_schema=False)
async def chat_and_shop() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/user-profile", include_in_schema=False)
async def user_profile(request: Request) -> Response:
    session = read_member_session(request)
    if session is None:
        return RedirectResponse("/?login=1", status_code=303)
    if session.get("stage") != "active":
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "user.html", headers={"Cache-Control": "no-store"})


@app.get("/user/{customer_no}", include_in_schema=False)
async def member_home(customer_no: str, request: Request) -> Response:
    session = read_member_session(request)
    if (
        not re.fullmatch(r"[1-9]\d{4}", customer_no)
        or session is None
        or session.get("stage") != "active"
        or session.get("cn") != customer_no
    ):
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "user.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
