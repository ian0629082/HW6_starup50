# GitHub Pages Deployment

這個專案的 `index.html` 已可用 GitHub Pages 靜態部署。

## 部署步驟

1. 將整個 `HW6` 資料夾推到 GitHub repository。
2. 到 GitHub repository 的 `Settings`。
3. 打開 `Pages`。
4. 在 `Build and deployment` 選擇：
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
5. 儲存後等待 GitHub 產生網址。

部署完成後，網址通常會是：

```text
https://你的帳號.github.io/你的repo名稱/
```

## 注意

GitHub Pages 不能執行 Python 後端，所以線上版不會使用 `server.py` 或 `best_startup_model.joblib`。  
`index.html` 已內建前端預測公式，因此別人點開 GitHub Pages 網址仍可直接操作滑桿並得到預測結果。

如果要使用真正的 Python 模型 API，需要把 `server.py` 部署到 Render、Railway、Fly.io 或其他可執行 Python 的平台，再把 `index.html` 的 `/predict` 改成該 API 網址。
