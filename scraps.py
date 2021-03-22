def save_to_text(result, num, ground_truth=None):
    if ground_truth is None:
        with open(f"{num}_{result}.json", 'w') as f:
            json.dump({num: result}, f)
        return f"{num}_{result}.json"
    else:
        with open(f"{num}_{result}_{ground_truth}.json", 'w') as f:
            json.dump({num: result, "ground_truth": ground_truth}, f)

@app.get("/")
def form_post(item: Item):
    return templates.TemplateResponse('download.html', context=item.dict())

@app.post('/')
def form_post(request: Request, item: Item):
    if action == 'submit':
        return templates.TemplateResponse('download.html', context=item.dict())
    elif action == 'submit and download':
        filepath = save_to_text(item.result, item.text)
        return FileResponse(filepath, media_type='application/octet-stream', filename='{}.json'.format(num))
    elif action == 'Yes':
        save_to_text(item.result, item.text, ground_truth=" Yes")
        return templates.TemplateResponse('download.html', context=item.dict())
    elif action == 'No':
        save_to_text(item.result, item.text, ground_truth=" No")
        return templates.TemplateResponse('download.html', context=item.dict())