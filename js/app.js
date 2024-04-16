const inputUsuario = document.getElementById('input-usuario');
const buttonPredecir = document.getElementById('button-predecir');
const outputDiv = document.getElementById('output');
const buttonEntrenar = document.getElementById('button-entrenar');
const spinner = document.getElementById('spinner');
const learnLinear = async () => {
    // Crear el modelo secuencial
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Datos de entrenamiento
    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

    // Entrenar el modelo
    await model.fit(xs, ys, {epochs: 250});

    // Indicar que el modelo está listo
    return model; // Devolver el modelo entrenado
};


// Al hacer clic en el botón, predecir Y para el valor de X ingresado por el usuario
buttonPredecir.addEventListener('click', async () => {
    const model = await learnLinear();
    const inputValue = parseFloat(inputUsuario.value);
    const output = model.predict(tf.tensor2d([inputValue], [1, 1]));
    const result = await output.data();
    outputDiv.textContent = `Para X = ${inputValue}, Y ≈ ${result[0]} redondeando seria: ${Math.round(result[0])}`;
});


buttonEntrenar.addEventListener('click', async () => {
    const model = await learnLinear();
    spinner.classList.remove('d-none');
    setTimeout(() => {
        spinner.classList.add('d-none');
    }, 2000);

});