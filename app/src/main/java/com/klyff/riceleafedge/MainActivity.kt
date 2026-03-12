package com.klyff.riceleafedge

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.klyff.riceleafedge.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var module: Module
    private lateinit var classes: List<String>

    private val takePicturePreview =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) {
                binding.previewImage.setImageBitmap(bitmap)
                runInference(bitmap)
            } else {
                binding.resultText.text = getString(R.string.capture_cancelled)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classes = assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        module = LiteModuleLoader.load(assetFilePath("rice_leaf_edge_mobile_int8.ptl"))

        binding.resultText.text = getString(R.string.ready_message)
        binding.captureButton.setOnClickListener {
            takePicturePreview.launch(null)
        }
    }

    private fun runInference(bitmap: Bitmap) {
        val preparedBitmap = centerCrop(scaleBitmap(bitmap, 144), 128)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            preparedBitmap,
            floatArrayOf(0.485f, 0.456f, 0.406f),
            floatArrayOf(0.229f, 0.224f, 0.225f)
        )

        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        val probabilities = softmax(scores)
        val bestIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[bestIndex] * 100f

        binding.resultText.text = getString(
            R.string.result_template,
            classes[bestIndex],
            String.format("%.1f", confidence)
        )
        binding.resultCard.setCardBackgroundColor(cardColorFor(classes[bestIndex]))
    }

    private fun softmax(values: FloatArray): FloatArray {
        val max = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size)
        var sum = 0.0
        for (i in values.indices) {
            exps[i] = kotlin.math.exp((values[i] - max).toDouble()).toFloat()
            sum += exps[i].toDouble()
        }
        return FloatArray(values.size) { i -> (exps[i] / sum.toFloat()) }
    }

    private fun scaleBitmap(bitmap: Bitmap, targetSize: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
    }

    private fun centerCrop(bitmap: Bitmap, size: Int): Bitmap {
        val xOffset = ((bitmap.width - size) / 2).coerceAtLeast(0)
        val yOffset = ((bitmap.height - size) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(bitmap, xOffset, yOffset, size, size)
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        assets.open(assetName).use { input ->
            file.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }

    private fun cardColorFor(label: String): Int {
        return if (label == "Healthy Rice Leaf") {
            Color.parseColor("#D7F0D1")
        } else {
            Color.parseColor("#F6D4CC")
        }
    }
}
