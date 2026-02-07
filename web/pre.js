var Module = {
    printErr: (text) => {
        if (arguments.length > 1)
            text = Array.prototype.slice.call(arguments).join(' ');
        console.debug(`[autoeq]%c  ${text}`, 'font-family: monospace;');
    }
};
