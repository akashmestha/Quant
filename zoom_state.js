window.addEventListener("plotly_relayout", (event) => {
    const detail = event.detail;
    if (detail && detail["xaxis.range[0]"]) {
        const payload = {
            x0: detail["xaxis.range[0]"],
            x1: detail["xaxis.range[1]"],
            y0: detail["yaxis.range[0]"],
            y1: detail["yaxis.range[1]"]
        };
        window.parent.postMessage({plotlyZoom: payload}, "*");
    }
});
