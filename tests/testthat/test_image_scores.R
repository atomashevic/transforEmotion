test_that("image_scores returns the correct output for a single face image", {
  # Test with example values
  trump = 'https://s.abcnews.com/images/US/trump-mugshot-main-ht-jt-230824_1692924861331_hpMain_16x9_1600.jpg'
  labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  result = image_scores(trump, labels)
  expect_equal(length(labels), ncol(result))
})

test_that("image_scores returns the correct output for a multiple face image", {
  # Test with example values
  image = 'https://cloudfront-us-east-1.images.arcpublishing.com/bostonglobe/BDI4NHFBKEYSK3GB34Q62HZJ4U.jpg'
  labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  result = image_scores(image, labels, face_selection = "largest")
  expect_equal(length(labels), ncol(result))
})

test_that("selecting left and right face produces different results",
{
  image = 'https://cloudfront-us-east-1.images.arcpublishing.com/bostonglobe/BDI4NHFBKEYSK3GB34Q62HZJ4U.jpg'
  labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  result_left = image_scores(image, labels, face_selection = "left")
  result_right = image_scores(image, labels, face_selection = "right")
  expect_equal(sum(result_left ==  result_right), 0 )
})