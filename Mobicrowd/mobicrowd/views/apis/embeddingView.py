# import json
# import base64
# import numpy as np
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from keras.models import load_model
# from sympy.codegen import Print
#
# # from mobicrowd.models.UNIQUE.ScoreModelUtils import load_score_model, evaluate
# import logging
#
# from PIL import Image
# import io
# # Define a score threshold
# SCORE_THRESHOLD = 0.75
#
# logger = logging.getLogger(__name__)
# def load_and_preprocess_embedding(request):
#     payload = json.loads(request.body)
#     encoded = payload.get('encoded', [])
#     if not isinstance(encoded, list):
#         raise ValueError("Encoded data should be a list")
#     encoded_array = np.array(encoded)
#     print(f"Encoded array shape: {encoded_array.shape}")
#     return encoded_array
#
#
# def classify_embedding(encoded_array):
#     classes = ['floods', 'fires', 'Other']
#     classification_model = load_model('mobicrowd/models/classif_model.h5')
#     predicted_class = classification_model.predict(encoded_array[:, :, :16, :])
#     predicted_class_index = np.argmax(predicted_class, axis=1)[0]
#     predicted_class = classes[int(predicted_class_index)]
#     return predicted_class
#
#
# def evaluate_image_quality(image_array):
#     score_model = load_score_model()
#     score, _ = evaluate(score_model, image_array)
#     return score
#
# def evaluate_original_image(image_base64):
#     try:
#         # Decode the base64 image data
#         image_data = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_data)).convert('RGB')
#         image_array = np.array(image)
#         print(f"Received image shape: {image_array.shape}")
#
#         # Evaluate the quality score
#         score = evaluate_image_quality(image_array)
#         print(f"Image score: {score}")
#
#         if score < SCORE_THRESHOLD:
#             return {
#                 'status': 'rejected',
#                 'message': 'Image quality is too low. Please submit a higher-quality image.',
#                 'score': score
#             }
#         else:
#             return {
#                 'status': 'accepted',
#                 'message': 'Image accepted.',
#                 'score': score,
#                 'image_base64': image_base64
#             }
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {'status': 'error', 'message': str(e)}
#
#
# def evaluate_embedding(embedding):
#     try:
#         logger.info(f"Accessing embedding element***********************************************")
#         # Wrap the call to classify_embedding in another try-except
#         try:
#             predicted_class = classify_embedding(np.array(embedding))
#
#         except Exception as e:
#             logger.error(f"Error inside classify_embedding: {str(e)}")
#             raise  # Re-raise the exception to be caught by the outer try-except
#
#         if predicted_class in ['floods', 'fires']:
#             verdict = True
#
#             logger.info(predicted_class)
#             return {
#                 'status': 'success',
#                 'predicted_class': predicted_class,
#                 'verdict': verdict,
#                 'message': 'Send original image for quality assessment.'
#             }
#         else:
#             logger.info(predicted_class)
#             verdict = False
#             return {
#                 'status': 'rejected',
#                 'predicted_class': predicted_class,
#                 'verdict': verdict,
#                 'message': 'Embedding does not belong to a desired class.'
#             }
#
#     except Exception as e:
#         logger.error(f"Error in evaluate_embedding: {str(e)}")
#         return {
#             'status': 'error',
#             'verdict': False,
#             'message': str(e)
#         }
